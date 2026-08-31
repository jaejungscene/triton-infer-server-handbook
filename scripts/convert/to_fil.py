#!/usr/bin/env python3
"""
to_fil.py — sklearn/XGBoost/LightGBM 모델을 FIL 백엔드 형식으로 변환

사용법:
    python scripts/convert/to_fil.py \
        --model-path ./weights/xgb_model.pkl \
        --output-path ./model_repository/anomaly_detector/1/xgboost.json \
        --format xgboost_json
"""

import argparse

from common import ConversionError, atomic_output, validated_input, validated_output


def convert_to_fil(
    model_path,
    output_path,
    fmt,
    expected_sha256=None,
    allow_unsafe_pickle=False,
):
    input_path = validated_input(model_path, expected_sha256)
    destination = validated_output(output_path, input_path)
    print(f"[convert] Loading from {input_path}")

    with atomic_output(destination) as temporary_output:
        if fmt == "xgboost_json":
            try:
                import xgboost as xgb
            except ImportError:
                raise ConversionError("xgboost is required") from None
            model = xgb.Booster()
            model.load_model(str(input_path))
            model.save_model(str(temporary_output))

        elif fmt == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ConversionError("lightgbm is required") from None
            model = lgb.Booster(model_file=str(input_path))
            model.save_model(str(temporary_output))

        else:
            if not allow_unsafe_pickle:
                raise ConversionError(
                    "treelite input uses pickle; pass --allow-unsafe-pickle only "
                    "for a trusted artifact"
                )
            try:
                import pickle
                import treelite
            except ImportError:
                raise ConversionError("treelite is required") from None
            print("[convert] WARNING: loading a trusted pickle checkpoint")
            with input_path.open("rb") as input_file:
                model = pickle.load(input_file)
            treelite_model = treelite.sklearn.import_model(model)
            treelite_model.serialize(str(temporary_output))

    print(f"[convert] Saved to {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--format", choices=["xgboost_json", "lightgbm", "treelite"], default="xgboost_json")
    parser.add_argument("--expected-sha256", help="Expected lowercase SHA-256 of the input")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow pickle loading for a trusted Treelite input",
    )
    args = parser.parse_args()
    try:
        convert_to_fil(
            args.model_path,
            args.output_path,
            args.format,
            args.expected_sha256,
            args.allow_unsafe_pickle,
        )
    except (ConversionError, OSError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
