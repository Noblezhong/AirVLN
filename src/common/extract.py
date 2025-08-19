import lmdb
import msgpack_numpy



def inspect_lmdb(lmdb_path, num_samples=1):
    """
    Inspect LMDB and print sample content safely.
    
    Args:
        lmdb_path (str): LMDB folder path
        num_samples (int): how many samples to inspect from the beginning
    """

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    with env.begin(write=False) as txn:
        cursor = txn.cursor()

        count = 0
        if cursor.first():
            while count < num_samples:
                key, value = cursor.item()
                print(f"\n===Sample key: {key.decode()} ===")

                sample = msgpack_numpy.unpackb(value, raw=False)
                print(f"Sample type: {type(sample)}")

                if isinstance(sample, list) and len(sample) == 3:
                    traj_obs, prev_actions, teacher_actions =sample
                    print("traj_obs type:", type(traj_obs))
                    if isinstance(traj_obs, dict):
                        for k, v in traj_obs.items():
                            if hasattr(v, "shape"):
                                print(f"{k}: (ndarray) shape={v.shape}, dtype={v.dtype}, first elements={v.flatten()[:5]}")
                            elif isinstance(v, list):
                                print(f"{k}: (list)len={len(v)}, first elements={v[:5]}")
                            elif isinstance(v, str):
                                print(f"{k}: (str) {v[:100]}...")
                            else:
                                print(f"{k}: {v}")
                        
                    print("prev_actions shape:", prev_actions.shape if hasattr(prev_actions, "shape") else None)
                    print("teacher_actions shape:", teacher_actions.shape if hasattr(teacher_actions, "shape") else None)
                
                elif hasattr(sample, "shape"):
                    print(f"ndarray shape={sample.shape}, dtype={sample.dtype}, first elements={sample.flatten()[:5]}")
                else:
                    print(sample)
                
                count += 1
                if not cursor.next():
                    break

# inspect_lmdb("../DATA/img_features/collect/AirVLN-seq2seq/train", num_samples=1)

# inspect_lmdb("../DATA/img_features/collect/AirVLN-seq2seq/train_rgb", num_samples=1)

# inspect_lmdb("../DATA/img_features/collect/AirVLN-seq2seq/train_depth", num_samples=1)