from typing import Any

def split_records_by_subset(records: list[dict[str, Any]]):
    """
    Based on roboflow dataset structure, splits records into train/val/test based on 'subset' field.
    """
    train_rec = [r for r in records if r.get("subset") == "train"]
    val_rec   = [r for r in records if r.get("subset") in ["valid", "val"]]
    test_rec  = [r for r in records if r.get("subset") == "test"]
    
    return train_rec, val_rec, test_rec