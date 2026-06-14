#!/usr/bin/env python3
import base64
import cgi
import csv
import io
import json
import math
import mimetypes
import os
import sys
import tempfile
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

APP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "ml-csv-lab-cache")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(APP_CACHE_DIR, "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(APP_CACHE_DIR, "xdg"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from sklearn.svm import SVC, SVR
except ImportError as exc:
    print("Missing dependency: {}".format(exc), file=sys.stderr)
    print("Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATASETS = {}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def safe_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def unique_columns(columns):
    seen = {}
    output = []
    for index, column in enumerate(columns):
        name = str(column).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = "Column {}".format(index + 1)
        count = seen.get(name, 0)
        seen[name] = count + 1
        output.append(name if count == 0 else "{} ({})".format(name, count + 1))
    return output


def detect_header(raw_bytes):
    sample = raw_bytes[:8192].decode("utf-8-sig", errors="replace")
    try:
        return bool(csv.Sniffer().has_header(sample))
    except csv.Error:
        rows = [row for row in csv.reader(io.StringIO(sample)) if row]
        if len(rows) < 2:
            return True

        def numeric_count(row):
            count = 0
            for item in row:
                try:
                    float(item)
                    count += 1
                except ValueError:
                    pass
            return count

        return numeric_count(rows[0]) < numeric_count(rows[1])


def parse_csv(raw_bytes, has_header):
    df = pd.read_csv(io.BytesIO(raw_bytes), header=0 if has_header else None)
    if df.empty:
        raise ValueError("The CSV file is empty.")
    df.columns = unique_columns(df.columns) if has_header else ["Column {}".format(i + 1) for i in range(len(df.columns))]
    return df


def infer_task_type(df, target_column):
    target = df[target_column].dropna()
    if target.empty:
        return "classification"
    numeric = pd.to_numeric(target, errors="coerce")
    unique_count = target.nunique(dropna=True)
    threshold = max(10, min(25, int(len(target) * 0.05)))
    return "regression" if numeric.notna().mean() > 0.95 and unique_count > threshold else "classification"


def dataset_summary(dataset_id):
    record = DATASETS[dataset_id]
    df = record["df"]
    target = df.columns[-1]
    preview = []
    for _, row in df.head(5).iterrows():
        preview.append({column: safe_value(row[column]) for column in df.columns})
    return {
        "dataset_id": dataset_id,
        "filename": record["filename"],
        "has_header": record["has_header"],
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "column_types": {column: str(df[column].dtype) for column in df.columns},
        "target_column": target,
        "task_type": infer_task_type(df, target),
        "preview": preview,
    }


def dataset_stats(df):
    numeric_columns = set(df.select_dtypes(include=[np.number]).columns)
    summaries = []
    distributions = []
    for column in df.columns:
        series = df[column]
        summary = {
            "column": column,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "mean": None,
            "std": None,
        }
        if column in numeric_columns:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric):
                summary["mean"] = safe_value(numeric.mean())
                summary["std"] = safe_value(numeric.std()) if len(numeric) > 1 else None
                counts, edges = np.histogram(numeric, bins=min(10, max(3, int(np.sqrt(len(numeric))))))
                distributions.append(
                    {
                        "column": column,
                        "kind": "numeric",
                        "labels": ["{:.3g} to {:.3g}".format(edges[i], edges[i + 1]) for i in range(len(counts))],
                        "values": [int(count) for count in counts],
                    }
                )
        else:
            counts = series.fillna("(missing)").astype(str).value_counts().head(10)
            distributions.append(
                {
                    "column": column,
                    "kind": "categorical",
                    "labels": list(counts.index),
                    "values": [int(value) for value in counts.values],
                }
            )
        summaries.append(summary)
    return {"summaries": summaries, "distributions": distributions[:12]}


def make_preprocessor(X, family):
    numeric_features = list(X.select_dtypes(include=[np.number]).columns)
    categorical_features = [column for column in X.columns if column not in numeric_features]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if family in {"linear", "svm"}:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers)


def get_model(task_type, model_name):
    regression = {
        "linear_regression": ("linear", LinearRegression()),
        "random_forest_regressor": ("tree", RandomForestRegressor(n_estimators=120, random_state=42)),
        "svr": ("svm", SVR()),
    }
    classification = {
        "logistic_regression": ("linear", LogisticRegression(max_iter=1200, class_weight="balanced")),
        "random_forest_classifier": ("tree", RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")),
        "svm": ("svm", SVC(class_weight="balanced")),
    }
    choices = regression if task_type == "regression" else classification
    if model_name not in choices:
        raise ValueError("Unsupported model selection.")
    return choices[model_name]


def plot_to_data_url():
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def train(df, target_column, task_type, model_name):
    if target_column not in df.columns:
        raise ValueError("Target column was not found.")
    if len(df) < 5:
        raise ValueError("At least five rows are required.")
    work_df = df.dropna(subset=[target_column]).copy()
    X = work_df.drop(columns=[target_column])
    y_raw = work_df[target_column]
    if X.empty:
        raise ValueError("At least one feature column is required.")

    family, estimator = get_model(task_type, model_name)
    preprocessor = make_preprocessor(X, family)

    if task_type == "regression":
        y = pd.to_numeric(y_raw, errors="coerce")
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        if len(y) < 5:
            raise ValueError("Regression requires at least five numeric target values.")
        stratify = None
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str).fillna("(missing)"))
        if len(encoder.classes_) < 2:
            raise ValueError("Classification requires at least two classes.")
        counts = pd.Series(y).value_counts()
        stratify = y if counts.min() >= 2 and len(y) >= len(counts) * 5 else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    if task_type == "regression":
        mse = mean_squared_error(y_test, predicted)
        metrics = {
            "MSE": safe_value(mse),
            "RMSE": safe_value(math.sqrt(mse)),
            "MAE": safe_value(mean_absolute_error(y_test, predicted)),
            "R2": safe_value(r2_score(y_test, predicted)) if len(y_test) > 1 else None,
        }
        plt.figure(figsize=(6.4, 4.2))
        plt.scatter(y_test, predicted, color="#0f766e", alpha=0.82, edgecolors="white", linewidth=0.5)
        lo = min(float(np.min(y_test)), float(np.min(predicted)))
        hi = max(float(np.max(y_test)), float(np.max(predicted)))
        plt.plot([lo, hi], [lo, hi], color="#c2410c", linewidth=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs. Predicted")
        plt.grid(True, alpha=0.25)
    else:
        labels = np.unique(np.concatenate([y_test, predicted]))
        matrix = confusion_matrix(y_test, predicted, labels=labels)
        class_names = encoder.inverse_transform(labels)
        metrics = {
            "Accuracy": safe_value(accuracy_score(y_test, predicted)),
            "Precision": safe_value(precision_score(y_test, predicted, average="weighted", zero_division=0)),
            "Recall": safe_value(recall_score(y_test, predicted, average="weighted", zero_division=0)),
            "F1": safe_value(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        }
        plt.figure(figsize=(6.2, 4.8))
        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar(fraction=0.046, pad=0.04)
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=35, ha="right")
        plt.yticks(ticks, class_names)
        threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "#17202a")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    return {
        "metrics": metrics,
        "visualization": plot_to_data_url(),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload, default=safe_value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"error": message}, status=status)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("Request is too large.")
        return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            path = "/index.html"
        file_path = os.path.join(STATIC_DIR, path[len("/static/") :]) if path.startswith("/static/") else os.path.join(BASE_DIR, path.lstrip("/"))
        file_path = os.path.abspath(file_path)
        if os.path.commonpath([BASE_DIR, file_path]) != BASE_DIR or not os.path.isfile(file_path):
            self.send_error_json("Not found.", status=404)
            return
        with open(file_path, "rb") as file:
            body = file.read()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file_path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            if self.path == "/api/upload":
                self.handle_upload()
            elif self.path == "/api/reparse":
                self.handle_reparse()
            elif self.path == "/api/stats":
                self.handle_stats()
            elif self.path == "/api/train":
                self.handle_train()
            else:
                self.send_error_json("Unknown endpoint.", status=404)
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(str(exc), status=400)

    def handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
        if "file" not in form:
            self.send_error_json("No file was uploaded.")
            return
        item = form["file"]
        filename = os.path.basename(item.filename or "")
        if not filename.lower().endswith(".csv"):
            self.send_error_json("Only CSV files are supported.", status=415)
            return
        raw_bytes = item.file.read()
        if not raw_bytes:
            self.send_error_json("The uploaded CSV is empty.")
            return
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            self.send_error_json("CSV files are limited to 20 MB.", status=413)
            return
        has_header = detect_header(raw_bytes)
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = {"filename": filename, "raw_bytes": raw_bytes, "has_header": has_header, "df": parse_csv(raw_bytes, has_header)}
        self.send_json(dataset_summary(dataset_id))

    def handle_reparse(self):
        payload = self.read_json()
        dataset_id = payload.get("dataset_id")
        if dataset_id not in DATASETS:
            self.send_error_json("Dataset was not found.", status=404)
            return
        DATASETS[dataset_id]["has_header"] = bool(payload.get("has_header"))
        DATASETS[dataset_id]["df"] = parse_csv(DATASETS[dataset_id]["raw_bytes"], DATASETS[dataset_id]["has_header"])
        self.send_json(dataset_summary(dataset_id))

    def handle_stats(self):
        payload = self.read_json()
        dataset_id = payload.get("dataset_id")
        if dataset_id not in DATASETS:
            self.send_error_json("Dataset was not found.", status=404)
            return
        self.send_json(dataset_stats(DATASETS[dataset_id]["df"]))

    def handle_train(self):
        payload = self.read_json()
        dataset_id = payload.get("dataset_id")
        if dataset_id not in DATASETS:
            self.send_error_json("Dataset was not found.", status=404)
            return
        self.send_json(train(DATASETS[dataset_id]["df"], payload.get("target_column"), payload.get("task_type"), payload.get("model_name")))


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = HTTPServer(("127.0.0.1", port), Handler)
    print("Running at http://127.0.0.1:{}".format(port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
