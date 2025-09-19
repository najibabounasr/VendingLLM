import pandas as pd

# this is the first version, has mistakes:
def load_and_clean_data_v1(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # Drop duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert date columns to datetime
    df["TransDate"] = pd.to_datetime(df["TransDate"], errors="coerce")
    df["Prcd Date"] = pd.to_datetime(df["Prcd Date"], errors="coerce")

    # Daily sales = sum(sales_per_product)
    # Create dataframe with 
    sales_per_product = (
        df.groupby(["Product", "TransDate"])
          .size()
          .reset_index(name="SalesCount")
    )
    # make daily sales. 
    daily_sales = (
        sales_per_product.groupby("TransDate")["SalesCount"]
        .sum()
        .reset_index()
    )
    
    daily_sales.rename(columns={"SalesCount": "TotalSales"}, inplace=True)

    # Average MPrice per day
    daily_avg_price = (
        df.groupby("TransDate")["MPrice"]
          .mean()
          .reset_index(name="AvgMPrice")
    )
    daily = pd.merge(daily_sales, daily_avg_price, on="TransDate")
    daily = daily.sort_values("TransDate")  # moved up, before any rolling/EMA/%change

    # Add features
    daily["Sales_Smoothed"] = daily["TotalSales"].rolling(window=7, center=True).mean()
    daily["EMA"] = daily["TotalSales"].ewm(span=14, adjust=False).mean()
    daily["Smoothed"] = daily["TotalSales"].rolling(window=7, center=False).mean()
    daily["Price_Smoothed"] = daily["AvgMPrice"].rolling(window=7, center=False).mean()
    daily["PricePctChange"] = daily["AvgMPrice"].pct_change() * 100

    # Date features
    daily["DayOfWeek"] = daily["TransDate"].dt.dayofweek
    daily["Month"] = daily["TransDate"].dt.month
    daily["Season"] = (daily["Month"] % 12 + 3) // 3  # 1=winter,2=spring,3=summer,4=fall

    return daily



def daily_sales(df):
    # Ensure types
    df = df.copy()
    df["TransDate"] = pd.to_datetime(df["TransDate"], errors="coerce").dt.date
    df = df.dropna(subset=["TransDate"])

    # Units = MQty (coerce safely)
    df["MQty"] = pd.to_numeric(df["MQty"], errors="coerce").fillna(0)

    # DAILY SALES (units) per date
    daily = (df.groupby("TransDate", as_index=False)["MQty"]
               .sum()
               .rename(columns={"MQty": "DailyUnits"}))
    
    daily = daily.sort_values("TransDate")  # NECCESSARY!

    # Optional price features (if needed)
    if "MPrice" in df.columns:
        df["MPrice"] = pd.to_numeric(df["MPrice"], errors="coerce")
        price_daily = (df.groupby("TransDate", as_index=False)["MPrice"]
                         .mean()
                         .rename(columns={"MPrice": "AvgMPrice"}))
        daily = daily.merge(price_daily, on="TransDate", how="left")
    

        daily["Price_Smoothed"] = daily["AvgMPrice"].rolling(7, center=False).mean()
        daily["PricePctChange"] = daily["AvgMPrice"].pct_change() * 100
    
    # Smoothing on units
    daily["EMA"] = daily["DailyUnits"].ewm(span=14, adjust=False).mean()
    daily["Sales_Smoothed"] = daily["DailyUnits"].rolling(7, center=False).mean()

    # Calendar features
    daily = daily.sort_values("TransDate")
    daily["TransDate"] = pd.to_datetime(daily["TransDate"])  # back to datetime for dt access
    daily["DayOfWeek"] = daily["TransDate"].dt.dayofweek
    daily["Month"] = daily["TransDate"].dt.month
    daily["Season"] = (daily["Month"] % 12 + 3) // 3  # 1..4

    return daily


