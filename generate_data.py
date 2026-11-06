
import pandas as pd
import numpy as np
import sqlite3
import os

from datetime import timedelta

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

SEED = 42
rng = np.random.default_rng(SEED)

OUTPUT_DIR = "inventory_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulation period
START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2026-10-31")

DATES = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="D"
)

# Dataset volume
N_PRODUCTS = 500
N_SUPPLIERS = 20
N_WAREHOUSES = 3

N_SALES = 100000
N_PURCHASE_ORDERS = 5000
N_DELIVERIES = 20000
N_MOVEMENTS = 150000

# Snapshot frequency (days)
SNAPSHOT_INTERVAL = 7


# ==========================================================
# 2. SUPPLIERS
# ==========================================================

regions = [
    "Southeast",
    "South",
    "Northeast",
    "Central-West",
    "North"
]

supplier_categories = [
    "Electronic Components",
    "IT Equipment",
    "Packaging",
    "Industrial Materials",
    "Accessories",
    "General Supplies"
]

suppliers = []

for i in range(N_SUPPLIERS):

    lead_time = int(rng.integers(3, 26))

    suppliers.append({
        "Supplier_ID": f"S{i+1:03d}",
        "Supplier_Name": f"Supplier {i+1:02d}",
        "Region": rng.choice(regions),
        "Contract_Lead_Time": lead_time,
        "Supplier_Category": rng.choice(
            supplier_categories
        )
    })

df_suppliers = pd.DataFrame(suppliers)


# ==========================================================
# 3. WAREHOUSES
# ==========================================================

warehouses = [
    {
        "Warehouse_ID": "WH01",
        "Warehouse_Name": "Sao Paulo Distribution Center",
        "Region": "Southeast"
    },
    {
        "Warehouse_ID": "WH02",
        "Warehouse_Name": "Curitiba Distribution Center",
        "Region": "South"
    },
    {
        "Warehouse_ID": "WH03",
        "Warehouse_Name": "Recife Distribution Center",
        "Region": "Northeast"
    }
]

df_warehouses = pd.DataFrame(warehouses)


# ==========================================================
# 4. PRODUCTS
# ==========================================================

categories = {
    "Peripherals": [
        "Wireless Mouse",
        "Mechanical Keyboard",
        "Headset",
        "Webcam",
        "Mouse Pad"
    ],
    "Storage": [
        "480GB SSD",
        "1TB SSD",
        "External Hard Drive",
        "64GB USB Flash Drive"
    ],
    "Hardware": [
        "RAM Memory",
        "Processor",
        "Graphics Card",
        "Motherboard"
    ],
    "Accessories": [
        "HDMI Cable",
        "Charger",
        "USB Hub",
        "USB Adapter"
    ],
    "Monitors": [
        "24-inch Monitor",
        "27-inch Monitor",
        "Ultrawide Monitor"
    ]
}

products = []

for i in range(N_PRODUCTS):

    category = rng.choice(list(categories.keys()))

    product_name = rng.choice(categories[category])

    unit_cost = round(
        float(rng.uniform(15, 1500)), 2
    )

    unit_price = round(
        unit_cost * rng.uniform(1.25, 2.0), 2
    )

    supplier_id = rng.choice(
        df_suppliers["Supplier_ID"].values
    )

    # Product demand classification
    demand_class = rng.choice(
        ["High", "Medium", "Low"],
        p=[0.20, 0.50, 0.30]
    )

    if demand_class == "High":
        daily_demand = rng.uniform(8, 25)

    elif demand_class == "Medium":
        daily_demand = rng.uniform(2, 8)

    else:
        daily_demand = rng.uniform(0.1, 2)

    # Product seasonality
    seasonality = rng.choice(
        [
            "Christmas",
            "Back to School",
            "Black Friday",
            "None"
        ],
        p=[0.25, 0.20, 0.20, 0.35]
    )

    safety_stock = int(daily_demand * 10)

    reorder_point = int(daily_demand * 20)

    target_stock = int(daily_demand * 45)

    products.append({
        "Product_ID": f"P{i+1:04d}",
        "Product_Name": product_name,
        "Category": category,
        "Supplier_ID": supplier_id,
        "Unit_Cost": unit_cost,
        "Unit_Price": unit_price,
        "Safety_Stock": safety_stock,
        "Reorder_Point": reorder_point,
        "Target_Stock": target_stock,
        "Demand_Class": demand_class,
        "Daily_Demand": round(
            float(daily_demand), 2
        ),
        "Seasonality": seasonality
    })

df_products = pd.DataFrame(products)

product_lookup = df_products.set_index("Product_ID")


# ==========================================================
# 5. SEASONAL DEMAND FUNCTION
# ==========================================================

def calculate_demand(product, date):

    demand = product["Daily_Demand"]

    month = date.month

    seasonality = product["Seasonality"]

    multiplier = 1.0

    if seasonality == "Christmas" and month in [11, 12]:
        multiplier = 2.8

    elif seasonality == "Back to School" and month in [1, 2]:
        multiplier = 2.5

    elif seasonality == "Black Friday" and month == 11:
        multiplier = 3.5

    quantity = rng.poisson(demand * multiplier)

    return max(0, int(quantity))


# ==========================================================
# 6. SALES
# ==========================================================

sales = []

product_ids = df_products["Product_ID"].values

sale_dates = rng.choice(
    DATES,
    size=N_SALES
)

for i in range(N_SALES):

    product_id = rng.choice(product_ids)

    product = product_lookup.loc[product_id]

    sale_date = pd.Timestamp(sale_dates[i])

    # Requested customer demand
    demand = calculate_demand(product, sale_date)

    demand = max(1, demand)

    # Simulated partial fulfillment
    fulfillment_rate = rng.choice(
        [1.0, 0.9, 0.7, 0.5],
        p=[0.82, 0.10, 0.05, 0.03]
    )

    fulfilled_quantity = int(
        demand * fulfillment_rate
    )

    unit_price = float(product["Unit_Price"])

    sales.append({
        "Sale_ID": f"SA{i+1:07d}",
        "Sale_Date": sale_date,
        "Product_ID": product_id,
        "Quantity": demand,
        "Unit_Price": unit_price,
        "Revenue": round(
            fulfilled_quantity * unit_price, 2
        ),
        "Fulfilled_Quantity": fulfilled_quantity
    })

df_sales = pd.DataFrame(sales)


# ==========================================================
# 7. PURCHASE ORDERS
# ==========================================================

purchase_orders = []

for i in range(N_PURCHASE_ORDERS):

    product = df_products.iloc[
        rng.integers(0, N_PRODUCTS)
    ]

    supplier_id = product["Supplier_ID"]

    supplier = df_suppliers[
        df_suppliers["Supplier_ID"] == supplier_id
    ].iloc[0]

    order_date = pd.Timestamp(
        rng.choice(DATES[:-30])
    )

    lead_time = int(
        supplier["Contract_Lead_Time"]
    )

    expected_date = order_date + timedelta(
        days=lead_time
    )

    # Simulate supplier delays
    delay = rng.choice(
        [0, 2, 5, 10, 15],
        p=[0.70, 0.12, 0.10, 0.06, 0.02]
    )

    actual_receipt_date = expected_date + timedelta(
        days=int(delay)
    )

    ordered_quantity = int(
        rng.integers(100, 1500)
    )

    # Simulate incomplete purchase orders
    receipt_rate = rng.choice(
        [1.0, 0.95, 0.80, 0.60],
        p=[0.85, 0.08, 0.05, 0.02]
    )

    received_quantity = int(
        ordered_quantity * receipt_rate
    )

    purchase_orders.append({
        "PO_ID": f"PO{i+1:06d}",
        "Supplier_ID": supplier_id,
        "Product_ID": product["Product_ID"],
        "Order_Date": order_date,
        "Expected_Date": expected_date,
        "Actual_Receipt_Date": actual_receipt_date,
        "Ordered_Quantity": ordered_quantity,
        "Received_Quantity": received_quantity
    })

df_purchase_orders = pd.DataFrame(purchase_orders)


# ==========================================================
# 8. DELIVERIES
# ==========================================================

deliveries = []

for i in range(N_DELIVERIES):

    promised_date = pd.Timestamp(
        rng.choice(DATES)
    )

    # Simulate early, on-time and late deliveries
    delivery_delay = rng.choice(
        [-2, 0, 1, 2, 5, 10],
        p=[0.05, 0.75, 0.08, 0.06, 0.04, 0.02]
    )

    actual_delivery_date = promised_date + timedelta(
        days=int(delivery_delay)
    )

    ordered_quantity = int(
        rng.integers(5, 300)
    )

    delivery_rate = rng.choice(
        [1.0, 0.95, 0.80, 0.60],
        p=[0.88, 0.06, 0.04, 0.02]
    )

    delivered_quantity = int(
        ordered_quantity * delivery_rate
    )

    deliveries.append({
        "Delivery_ID": f"DL{i+1:07d}",
        "Order_ID": f"SO{i+1:07d}",
        "Promised_Date": promised_date,
        "Actual_Delivery_Date": actual_delivery_date,
        "Ordered_Quantity": ordered_quantity,
        "Delivered_Quantity": delivered_quantity
    })

df_deliveries = pd.DataFrame(deliveries)


# ==========================================================
# 9. STOCK MOVEMENTS
# ==========================================================

movements = []

movement_types = [
    "Sale",
    "Purchase",
    "Return",
    "Adjustment"
]

for i in range(N_MOVEMENTS):

    product_id = rng.choice(product_ids)

    product = product_lookup.loc[product_id]

    movement_date = pd.Timestamp(
        rng.choice(DATES)
    )

    movement_type = rng.choice(
        movement_types,
        p=[0.55, 0.30, 0.10, 0.05]
    )

    quantity = int(
        rng.integers(1, 100)
    )

    # Positive quantities represent stock inflows
    # Negative quantities represent stock outflows
    if movement_type in ["Sale", "Adjustment"]:
        quantity = -quantity

    unit_cost = float(product["Unit_Cost"])

    movements.append({
        "Movement_ID": f"MV{i+1:07d}",
        "Movement_Date": movement_date,
        "Product_ID": product_id,
        "Movement_Type": movement_type,
        "Quantity": quantity,
        "Unit_Cost": unit_cost
    })

df_movements = pd.DataFrame(movements)


# ==========================================================
# 10. HISTORICAL INVENTORY SNAPSHOTS
# ==========================================================

snapshot_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq=f"{SNAPSHOT_INTERVAL}D"
)

if snapshot_dates[-1] != END_DATE:

    snapshot_dates = snapshot_dates.append(
        pd.DatetimeIndex([END_DATE])
    )

snapshots = []

# Initialize stock for each product and warehouse
stock_state = {}

for product_id in product_ids:

    product = product_lookup.loc[product_id]

    for warehouse_id in df_warehouses["Warehouse_ID"]:

        initial_stock = int(
            product["Target_Stock"] *
            rng.uniform(0.5, 1.3)
        )

        stock_state[
            (product_id, warehouse_id)
        ] = initial_stock


for date in snapshot_dates:

    for product_id in product_ids:

        product = product_lookup.loc[product_id]

        total_demand = calculate_demand(
            product, date
        )

        for warehouse_id in df_warehouses["Warehouse_ID"]:

            key = (product_id, warehouse_id)

            current_stock = stock_state[key]

            warehouse_demand = int(
                total_demand / N_WAREHOUSES
            )

            # Replenish stock when it reaches the reorder point
            if current_stock < product["Reorder_Point"]:

                replenishment = int(
                    rng.uniform(
                        product["Safety_Stock"],
                        product["Target_Stock"]
                    )
                )

                current_stock += replenishment

            # Simulate reserved inventory
            reserved_stock = int(
                current_stock * rng.uniform(0.02, 0.15)
            )

            reserved_stock = min(
                reserved_stock,
                current_stock
            )

            available_stock = (
                current_stock - reserved_stock
            )

            snapshots.append({
                "Snapshot_Date": date,
                "Product_ID": product_id,
                "Warehouse_ID": warehouse_id,
                "Stock_On_Hand": current_stock,
                "Stock_Reserved": reserved_stock,
                "Stock_Available": available_stock,
                "Inventory_Value": round(
                    current_stock * product["Unit_Cost"],
                    2
                )
            })

            # Update stock for the next snapshot
            current_stock = max(
                0,
                current_stock - warehouse_demand
            )

            stock_state[key] = current_stock


df_snapshots = pd.DataFrame(snapshots)


# ==========================================================
# 11. EXPORT CSV FILES
# ==========================================================

tables = {
    "Products": df_products,
    "Suppliers": df_suppliers,
    "Warehouses": df_warehouses,
    "Sales": df_sales,
    "Purchase_Orders": df_purchase_orders,
    "Deliveries": df_deliveries,
    "Stock_Movements": df_movements,
    "Inventory_Snapshot": df_snapshots
}

for table_name, dataframe in tables.items():

    file_path = os.path.join(
        OUTPUT_DIR,
        f"{table_name}.csv"
    )

    dataframe.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"{table_name}: {len(dataframe):,} records generated"
    )


# ==========================================================
# 12. SQLITE DATABASE
# ==========================================================

database_path = os.path.join(
    OUTPUT_DIR,
    "inventory_dataset.db"
)

with sqlite3.connect(database_path) as connection:

    for table_name, dataframe in tables.items():

        dataframe.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False
        )


# ==========================================================
# 13. DATA VALIDATION
# ==========================================================

print("\n========== DATA VALIDATION ==========")

# Check primary keys
assert df_products["Product_ID"].is_unique
assert df_suppliers["Supplier_ID"].is_unique
assert df_sales["Sale_ID"].is_unique
assert df_purchase_orders["PO_ID"].is_unique

# Check foreign keys
assert set(
    df_sales["Product_ID"]
).issubset(
    set(df_products["Product_ID"])
)

assert set(
    df_products["Supplier_ID"]
).issubset(
    set(df_suppliers["Supplier_ID"])
)

# Check inventory calculations
assert (
    df_snapshots["Stock_Available"]
    ==
    df_snapshots["Stock_On_Hand"]
    - df_snapshots["Stock_Reserved"]
).all()

# Check sales fulfillment
assert (
    df_sales["Fulfilled_Quantity"]
    <= df_sales["Quantity"]
).all()

# Check purchase order quantities
assert (
    df_purchase_orders["Received_Quantity"]
    <= df_purchase_orders["Ordered_Quantity"]
).all()

# Check delivery quantities
assert (
    df_deliveries["Delivered_Quantity"]
    <= df_deliveries["Ordered_Quantity"]
).all()

print("All basic data validation checks passed.")


# ==========================================================
# 14. FINAL SUMMARY
# ==========================================================

print("\n========== DATASET SUMMARY ==========")

for table_name, dataframe in tables.items():

    print(
        f"{table_name}: {len(dataframe):,} records"
    )

print("\nOutput directory:", OUTPUT_DIR)

print("SQLite database:", database_path)

print("\n========== DATASET GENERATION COMPLETED ==========")
