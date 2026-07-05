"""
train_models.py
----------------
Runs the full pipeline end to end, in order:
    1. data_cleaning
    2. eda
    3. rfm_segmentation
    4. recommendation

Use this when you just want to (re)build every artifact the Streamlit
app needs in one go.

Run directly:
    python src/train_models.py
"""

import data_cleaning
import eda
import rfm_segmentation
import recommendation


def main():
    print("=" * 70)
    print("STEP 1/4: Data Cleaning")
    print("=" * 70)
    data_cleaning.main()

    print("\n" + "=" * 70)
    print("STEP 2/4: Exploratory Data Analysis")
    print("=" * 70)
    eda.main()

    print("\n" + "=" * 70)
    print("STEP 3/4: RFM Customer Segmentation")
    print("=" * 70)
    rfm_segmentation.main()

    print("\n" + "=" * 70)
    print("STEP 4/4: Product Recommendation Model")
    print("=" * 70)
    recommendation.main()

    print("\nAll done! Launch the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
