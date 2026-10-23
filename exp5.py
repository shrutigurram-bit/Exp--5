import pandas as pd
from datetime import datetime


class SimpleFeatureStore:
    def __init__(self):
        # Repository metadata: registered feature definitions
        self.repository = {}

        # Offline store keeps historical/tabular feature records
        self.offline_store = {}

        # Online store keeps the latest values for each entity
        self.online_store = {}

    def register_feature(self, name, entity_key, columns, description):
        """Register a feature group and its schema in the repository."""
        if name in self.repository:
            raise ValueError(
                f"Feature group '{name}' is already registered."
            )

        self.repository[name] = {
            "entity_key": entity_key,
            "columns": columns,
            "description": description,
            "registered_at": datetime.now().isoformat(
                timespec="seconds"
            )
        }

        print(f"Registered feature group: {name}")

    def ingest(self, name, dataframe):
        """Validate a DataFrame and write it to offline and online stores."""
        if name not in self.repository:
            raise ValueError(
                f"Register '{name}' before ingestion."
            )

        metadata = self.repository[name]
        required = [metadata["entity_key"]] + metadata["columns"]

        missing = [
            col for col in required
            if col not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        data = dataframe[required].copy()

        if data[metadata["entity_key"]].isna().any():
            raise ValueError(
                "Entity key cannot contain missing values."
            )

        # Offline: retain all ingested rows for training/analysis
        if name not in self.offline_store:
            self.offline_store[name] = data.copy()
        else:
            self.offline_store[name] = pd.concat(
                [self.offline_store[name], data],
                ignore_index=True
            )

        # Online: latest row for each entity (simplified demonstration)
        latest = data.drop_duplicates(
            subset=[metadata["entity_key"]],
            keep="last"
        )

        if name not in self.online_store:
            self.online_store[name] = {}

        for _, row in latest.iterrows():
            key = row[metadata["entity_key"]]
            self.online_store[name][key] = row.to_dict()

        print(f"Ingested {len(data)} rows into '{name}'.")

    def get_offline_features(self, name, entity_ids=None):
        """Return historical data for model training and analysis."""
        if name not in self.offline_store:
            raise ValueError(
                f"No offline data found for '{name}'."
            )

        data = self.offline_store[name].copy()

        key = self.repository[name]["entity_key"]

        if entity_ids is not None:
            data = data[data[key].isin(entity_ids)]
            data = data.reset_index(drop=True)

        return data

    def get_online_features(self, name, entity_id):
        """Return the latest feature values for one entity."""
        if name not in self.online_store:
            raise ValueError(
                f"No online data found for '{name}'."
            )

        return self.online_store[name].get(entity_id)

    def show_repository(self):
        return pd.DataFrame(self.repository).T


# 1. Create the feature store / repository
store = SimpleFeatureStore()


# 2. Define and register reusable customer features
store.register_feature(
    name="customer_features",
    entity_key="customer_id",
    columns=[
        "total_orders",
        "avg_order_value",
        "days_since_last_order"
    ],
    description="Purchase activity features for customer models"
)


# 3. Prepare example data and ingest it
customer_data = pd.DataFrame({
    "customer_id": [101, 102, 103, 101],
    "total_orders": [5, 2, 9, 6],
    "avg_order_value": [450.0, 220.0, 180.0, 470.0],
    "days_since_last_order": [4, 20, 2, 1]
})

store.ingest("customer_features", customer_data)


# 4. Inspect registered definitions
print("\nFEATURE REPOSITORY")
print(
    store.show_repository()[
        ["entity_key", "columns", "description"]
    ]
)


# 5. Retrieve offline features for training
print("\nOFFLINE FEATURES FOR TRAINING")
training_data = store.get_offline_features(
    "customer_features",
    entity_ids=[101, 102, 103]
)

print(training_data)


# 6. Retrieve latest online features for inference
print("\nONLINE FEATURES FOR CUSTOMER 101")
online_features = store.get_online_features(
    "customer_features", 101
)

print(online_features)


# 7. Reuse the same features for two example model workflows

# Model A: customer churn risk model
churn_input = online_features
print("\nMODEL A (CHURN) INPUT:")
print(churn_input)

# Model B: customer offer recommendation model
offer_input = online_features
print("\nMODEL B (OFFER RECOMMENDATION) INPUT:")
print(offer_input)