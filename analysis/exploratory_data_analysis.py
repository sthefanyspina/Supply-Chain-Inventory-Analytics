# ==========================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# INVENTORY DATASET PROJECT
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ==========================================================
# LOAD DATA
# ==========================================================

DATA_PATH = "inventory_dataset"

products = pd.read_csv(f"{DATA_PATH}/Products.csv")
suppliers = pd.read_csv(f"{DATA_PATH}/Suppliers.csv")
warehouses = pd.read_csv(f"{DATA_PATH}/Warehouses.csv")
sales = pd.read_csv(f"{DATA_PATH}/Sales.csv")
purchase_orders = pd.read_csv(f"{DATA_PATH}/Purchase_Orders.csv")
deliveries = pd.read_csv(f"{DATA_PATH}/Deliveries.csv")
movements = pd.read_csv(f"{DATA_PATH}/Stock_Movements.csv")
inventory = pd.read_csv(f"{DATA_PATH}/Inventory_Snapshot.csv")

# ==========================================================
# DATE FORMATTING
# ==========================================================

sales["Sale_Date"] = pd.to_datetime(sales["Sale_Date"])

purchase_orders["Order_Date"] = pd.to_datetime(
    purchase_orders["Order_Date"]
)

purchase_orders["Expected_Date"] = pd.to_datetime(
    purchase_orders["Expected_Date"]
)

purchase_orders["Actual_Receipt_Date"] = pd.to_datetime(
    purchase_orders["Actual_Receipt_Date"]
)

inventory["Snapshot_Date"] = pd.to_datetime(
    inventory["Snapshot_Date"]
)

deliveries["Promised_Date"] = pd.to_datetime(
    deliveries["Promised_Date"]
)

deliveries["Actual_Delivery_Date"] = pd.to_datetime(
    deliveries["Actual_Delivery_Date"]
)

# ==========================================================
# LAST SNAPSHOT
# ==========================================================

last_snapshot_date = inventory["Snapshot_Date"].max()

latest_inventory = inventory[
    inventory["Snapshot_Date"] == last_snapshot_date
].copy()

latest_inventory = latest_inventory.merge(
    products,
    on="Product_ID",
    how="left"
)

latest_inventory = latest_inventory.merge(
    warehouses,
    on="Warehouse_ID",
    how="left"
)

print("\n")
print("="*80)
print("A. INVENTORY ANALYSIS")
print("="*80)

# ==========================================================
# A1. PRODUCTS WITH HIGHEST STOCK
# ==========================================================

high_stock = (
    latest_inventory
    .groupby(["Product_ID","Product_Name"])
    ["Stock_On_Hand"]
    .sum()
    .reset_index()
    .sort_values("Stock_On_Hand", ascending=False)
)

print("\nTOP 10 PRODUCTS BY STOCK")
print(high_stock.head(10))

# ==========================================================
# A2. PRODUCTS BELOW SAFETY STOCK
# ==========================================================

below_safety = latest_inventory[
    latest_inventory["Stock_On_Hand"]
    <
    latest_inventory["Safety_Stock"]
]

below_safety_summary = (
    below_safety
    .groupby(["Product_ID","Product_Name"])
    ["Stock_On_Hand"]
    .sum()
    .reset_index()
)

print("\nPRODUCTS BELOW SAFETY STOCK")
print(below_safety_summary.head(20))

print(
    f"\nNumber of critical products: "
    f"{below_safety_summary['Product_ID'].nunique()}"
)

# ==========================================================
# A3. EXCESS INVENTORY
# ==========================================================

excess_inventory = latest_inventory[
    latest_inventory["Stock_On_Hand"]
    >
    latest_inventory["Target_Stock"]
]

excess_summary = (
    excess_inventory
    .groupby(["Product_ID","Product_Name"])
    ["Stock_On_Hand"]
    .sum()
    .reset_index()
)

print("\nPRODUCTS WITH EXCESS INVENTORY")
print(excess_summary.head(20))

# ==========================================================
# A4. WAREHOUSE WITH HIGHEST STOCK VALUE
# ==========================================================

warehouse_value = (
    latest_inventory
    .groupby("Warehouse_Name")
    ["Inventory_Value"]
    .sum()
    .sort_values(ascending=False)
)

print("\nINVENTORY VALUE BY WAREHOUSE")
print(warehouse_value)

# ==========================================================
# A5. INVENTORY TREND
# ==========================================================

inventory_evolution = (
    inventory
    .groupby("Snapshot_Date")
    ["Inventory_Value"]
    .sum()
)

plt.figure(figsize=(14,6))

inventory_evolution.plot()

plt.title("Inventory Value Evolution")
plt.ylabel("Inventory Value")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# ==========================================================
# SALES & DEMAND ANALYSIS
# ==========================================================

print("\n")
print("="*80)
print("B. SALES & DEMAND ANALYSIS")
print("="*80)

sales = sales.merge(
    products[
        [
            "Product_ID",
            "Product_Name",
            "Category"
        ]
    ],
    on="Product_ID",
    how="left"
)

# ==========================================================
# B1. HIGHEST DEMAND PRODUCTS
# ==========================================================

highest_demand = (
    sales
    .groupby(["Product_ID","Product_Name"])
    ["Quantity"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP 10 PRODUCTS BY DEMAND")
print(highest_demand.head(10))

# ==========================================================
# B2. LOWEST DEMAND PRODUCTS
# ==========================================================

lowest_demand = (
    sales
    .groupby(["Product_ID","Product_Name"])
    ["Quantity"]
    .sum()
    .sort_values(ascending=True)
)

print("\nLOWEST DEMAND PRODUCTS")
print(lowest_demand.head(10))

# ==========================================================
# B3. MONTHLY SALES EVOLUTION
# ==========================================================

sales["YearMonth"] = (
    sales["Sale_Date"]
    .dt.to_period("M")
    .astype(str)
)

monthly_sales = (
    sales
    .groupby("YearMonth")
    ["Revenue"]
    .sum()
)

plt.figure(figsize=(16,6))

monthly_sales.plot()

plt.title("Monthly Revenue Evolution")
plt.ylabel("Revenue")
plt.xlabel("Month")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# ==========================================================
# B4. PARTIAL FULFILLMENT
# ==========================================================

sales["Fill_Rate"] = (
    sales["Fulfilled_Quantity"]
    /
    sales["Quantity"]
)

partial_orders = sales[
    sales["Fill_Rate"] < 1
]

print(
    f"\nPartial fulfillment sales: "
    f"{len(partial_orders):,}"
)

print(
    f"Service level: "
    f"{sales['Fill_Rate'].mean():.2%}"
)

# ==========================================================
# B5. REVENUE BY CATEGORY
# ==========================================================

category_revenue = (
    sales
    .groupby("Category")
    ["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\nREVENUE BY CATEGORY")
print(category_revenue)

plt.figure(figsize=(10,6))

category_revenue.plot(
    kind="bar"
)

plt.title("Revenue by Category")

plt.tight_layout()
plt.show()

# ==========================================================
# SUPPLIER ANALYSIS
# ==========================================================

print("\n")
print("="*80)
print("C. SUPPLIER ANALYSIS")
print("="*80)

supplier_orders = (
    purchase_orders
    .merge(
        suppliers,
        on="Supplier_ID",
        how="left"
    )
)

# ==========================================================
# C1. HIGHEST LEAD TIME
# ==========================================================

leadtime_analysis = (
    suppliers
    .sort_values(
        "Contract_Lead_Time",
        ascending=False
    )
)

print("\nSUPPLIERS WITH HIGHEST LEAD TIME")
print(
    leadtime_analysis[
        [
            "Supplier_Name",
            "Contract_Lead_Time"
        ]
    ].head(10)
)

# ==========================================================
# C2. DELAYED PURCHASE ORDERS
# ==========================================================

supplier_orders["Delay_Days"] = (
    supplier_orders["Actual_Receipt_Date"]
    -
    supplier_orders["Expected_Date"]
).dt.days

late_pos = supplier_orders[
    supplier_orders["Delay_Days"] > 0
]

print(
    f"\nLate purchase orders: "
    f"{len(late_pos):,}"
)

supplier_delay = (
    late_pos
    .groupby("Supplier_Name")
    ["Delay_Days"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAVERAGE DELAY BY SUPPLIER")
print(supplier_delay.head(10))

# ==========================================================
# C3. INCOMPLETE ORDERS
# ==========================================================

incomplete_orders = supplier_orders[
    supplier_orders["Received_Quantity"]
    <
    supplier_orders["Ordered_Quantity"]
]

print(
    f"\nIncomplete Purchase Orders: "
    f"{len(incomplete_orders):,}"
)

# ==========================================================
# C4. PURCHASE VOLUME
# ==========================================================

purchase_volume = (
    supplier_orders
    .groupby("Supplier_Name")
    ["Ordered_Quantity"]
    .sum()
    .sort_values(ascending=False)
)

print("\nTOP SUPPLIERS BY PURCHASE VOLUME")
print(purchase_volume.head(10))

# ==========================================================
# WAREHOUSE ANALYSIS
# ==========================================================

print("\n")
print("="*80)
print("D. WAREHOUSE ANALYSIS")
print("="*80)

# ==========================================================
# D1. STOCK BY WAREHOUSE
# ==========================================================

stock_by_warehouse = (
    latest_inventory
    .groupby("Warehouse_Name")
    ["Stock_On_Hand"]
    .sum()
)

print("\nTOTAL STOCK BY WAREHOUSE")
print(stock_by_warehouse)

# ==========================================================
# D2. INVENTORY VALUE
# ==========================================================

inventory_value_wh = (
    latest_inventory
    .groupby("Warehouse_Name")
    ["Inventory_Value"]
    .sum()
)

print("\nINVENTORY VALUE BY WAREHOUSE")
print(inventory_value_wh)

# ==========================================================
# D3. CRITICAL PRODUCTS
# ==========================================================

critical_products = (
    latest_inventory
    .assign(
        Critical=
        latest_inventory["Stock_On_Hand"]
        <
        latest_inventory["Safety_Stock"]
    )
)

critical_by_wh = (
    critical_products
    .groupby("Warehouse_Name")
    ["Critical"]
    .sum()
)

print("\nCRITICAL PRODUCTS BY WAREHOUSE")
print(critical_by_wh)

# ==========================================================
# D4. AVAILABLE STOCK DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    data=latest_inventory,
    x="Warehouse_Name",
    y="Stock_Available"
)

plt.title(
    "Available Stock Distribution by Warehouse"
)

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

# ==========================================================
# OPERATIONAL ISSUES FOR DASHBOARD
# ==========================================================

print("\n")
print("="*80)
print("MAIN OPERATIONAL ISSUES IDENTIFIED")
print("="*80)

print("""
1. Products below safety stock
   -> Risk of stockout.

2. Products above target stock
   -> Excess working capital.

3. Partial fulfillment orders
   -> Service level reduction.

4. Suppliers with frequent delays
   -> Supply chain instability.

5. Suppliers delivering incomplete quantities
   -> Inventory replenishment failures.

6. Warehouses concentrating most inventory value
   -> Capital concentration risk.

7. Categories with highest demand growth
   -> Replenishment priority.

8. Significant imbalance between warehouses
   -> Need for stock redistribution.

9. Seasonal sales peaks
   -> Demand planning opportunity.

10. Products generating high revenue but low inventory
    -> Immediate replenishment candidates.
""")
