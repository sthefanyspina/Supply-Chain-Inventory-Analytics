# ============================================================
# SUPPLY CHAIN & INVENTORY ANALYTICS
# EXPLORATORY DATA ANALYSIS (EDA)
#
# Areas:
# A. Inventory Analysis
# B. Sales & Demand Analysis
# C. Supplier Analysis
# D. Warehouse Analysis
#
# Outputs:
# - Analytical CSV files
# - Data visualizations (PNG)
# - Executive report (TXT)
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path("inventory_dataset")

OUTPUT_DIR = BASE_DIR / "analysis"

INVENTORY_DIR = OUTPUT_DIR / "inventory_analysis"
SALES_DIR = OUTPUT_DIR / "sales_demand_analysis"
SUPPLIER_DIR = OUTPUT_DIR / "supplier_analysis"
WAREHOUSE_DIR = OUTPUT_DIR / "warehouse_analysis"
REPORT_DIR = OUTPUT_DIR / "executive_report"

FOLDERS = [
    INVENTORY_DIR,
    SALES_DIR,
    SUPPLIER_DIR,
    WAREHOUSE_DIR,
    REPORT_DIR
]

for folder in FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

REQUIRED_FILES = [
    "Products.csv",
    "Suppliers.csv",
    "Warehouses.csv",
    "Sales.csv",
    "Purchase_Orders.csv",
    "Deliveries.csv",
    "Stock_Movements.csv",
    "Inventory_Snapshot.csv"
]

for filename in REQUIRED_FILES:
    if not (BASE_DIR / filename).exists():
        raise FileNotFoundError(
            f"Dataset not found: {BASE_DIR / filename}"
        )

products = pd.read_csv(
    BASE_DIR / "Products.csv"
)

suppliers = pd.read_csv(
    BASE_DIR / "Suppliers.csv"
)

warehouses = pd.read_csv(
    BASE_DIR / "Warehouses.csv"
)

sales = pd.read_csv(
    BASE_DIR / "Sales.csv",
    parse_dates=["Sale_Date"]
)

purchase_orders = pd.read_csv(
    BASE_DIR / "Purchase_Orders.csv",
    parse_dates=[
        "Order_Date",
        "Expected_Date",
        "Actual_Receipt_Date"
    ]
)

deliveries = pd.read_csv(
    BASE_DIR / "Deliveries.csv",
    parse_dates=[
        "Promised_Date",
        "Actual_Delivery_Date"
    ]
)

stock_movements = pd.read_csv(
    BASE_DIR / "Stock_Movements.csv",
    parse_dates=["Movement_Date"]
)

inventory = pd.read_csv(
    BASE_DIR / "Inventory_Snapshot.csv",
    parse_dates=["Snapshot_Date"]
)

print("Datasets loaded successfully.")


# ============================================================
# 4. DATA QUALITY VALIDATION
# ============================================================

print("\nValidating data quality...")

datasets = {
    "Products": products,
    "Suppliers": suppliers,
    "Warehouses": warehouses,
    "Sales": sales,
    "Purchase_Orders": purchase_orders,
    "Deliveries": deliveries,
    "Stock_Movements": stock_movements,
    "Inventory_Snapshot": inventory
}

quality_report = []

for name, df in datasets.items():

    quality_report.append({
        "Dataset": name,
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate_Rows": df.duplicated().sum(),
        "Missing_Values": df.isna().sum().sum()
    })

quality_report = pd.DataFrame(quality_report)

quality_report.to_csv(
    REPORT_DIR / "data_quality_report.csv",
    index=False
)

print(quality_report.to_string(index=False))


# ============================================================
# 5. DATA PREPARATION
# ============================================================

print("\nPreparing datasets...")

# Convert numeric columns

numeric_columns = {
    "Products": [
        "Unit_Cost",
        "Unit_Price",
        "Safety_Stock",
        "Reorder_Point",
        "Target_Stock"
    ],
    "Sales": [
        "Quantity",
        "Fulfilled_Quantity",
        "Unit_Price",
        "Revenue"
    ],
    "Inventory": [
        "Stock_On_Hand",
        "Stock_Reserved",
        "Stock_Available",
        "Inventory_Value"
    ],
    "Purchase_Orders": [
        "Ordered_Quantity",
        "Received_Quantity"
    ],
    "Deliveries": [
        "Ordered_Quantity",
        "Delivered_Quantity"
    ]
}

dataframes = {
    "Products": products,
    "Sales": sales,
    "Inventory": inventory,
    "Purchase_Orders": purchase_orders,
    "Deliveries": deliveries
}

for dataframe_name, columns in numeric_columns.items():

    df = dataframes[dataframe_name]

    for column in columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# Recalculate available stock
# This validates the physical and reserved quantities.

inventory["Calculated_Available_Stock"] = (
    inventory["Stock_On_Hand"]
    - inventory["Stock_Reserved"]
)

inventory["Stock_Consistency_Check"] = (
    inventory["Calculated_Available_Stock"]
    == inventory["Stock_Available"]
)

# Calculate sales fulfillment

sales["Unfulfilled_Quantity"] = (
    sales["Quantity"]
    - sales["Fulfilled_Quantity"]
).clip(lower=0)

sales["Is_Partial_Fulfillment"] = (
    sales["Fulfilled_Quantity"] < sales["Quantity"]
)

# Calculate purchase order lead time

purchase_orders["Actual_Lead_Time"] = (
    purchase_orders["Actual_Receipt_Date"]
    - purchase_orders["Order_Date"]
).dt.days

purchase_orders["Expected_Lead_Time"] = (
    purchase_orders["Expected_Date"]
    - purchase_orders["Order_Date"]
).dt.days

purchase_orders["Delay_Days"] = (
    purchase_orders["Actual_Receipt_Date"]
    - purchase_orders["Expected_Date"]
).dt.days

purchase_orders["Is_Late"] = (
    purchase_orders["Actual_Receipt_Date"]
    > purchase_orders["Expected_Date"]
)

purchase_orders["Is_Incomplete"] = (
    purchase_orders["Received_Quantity"]
    < purchase_orders["Ordered_Quantity"]
)

purchase_orders["Missing_Quantity"] = (
    purchase_orders["Ordered_Quantity"]
    - purchase_orders["Received_Quantity"]
).clip(lower=0)

# Calculate delivery performance

deliveries["Is_On_Time"] = (
    deliveries["Actual_Delivery_Date"]
    <= deliveries["Promised_Date"]
)

deliveries["Is_In_Full"] = (
    deliveries["Delivered_Quantity"]
    >= deliveries["Ordered_Quantity"]
)

deliveries["Is_OTIF"] = (
    deliveries["Is_On_Time"]
    & deliveries["Is_In_Full"]
)

# Add product information

inventory = inventory.merge(
    products[
        [
            "Product_ID",
            "Product_Name",
            "Category",
            "Supplier_ID",
            "Unit_Cost",
            "Safety_Stock",
            "Reorder_Point",
            "Target_Stock"
        ]
    ],
    on="Product_ID",
    how="left"
)

sales = sales.merge(
    products[
        [
            "Product_ID",
            "Product_Name",
            "Category",
            "Unit_Cost"
        ]
    ],
    on="Product_ID",
    how="left"
)

purchase_orders = purchase_orders.merge(
    suppliers[
        [
            "Supplier_ID",
            "Supplier_Name",
            "Region",
            "Contract_Lead_Time"
        ]
    ],
    on="Supplier_ID",
    how="left"
)

inventory = inventory.merge(
    warehouses[
        [
            "Warehouse_ID",
            "Warehouse_Name"
        ]
    ],
    on="Warehouse_ID",
    how="left"
)

print("Data preparation completed.")

