#!/usr/bin/env python3
"""
Dataset Metadata Fixer for LeRobot
===================================
Bu script, LeRobot dataset'inin metadata dosyalarını düzeltir ve
eğitime hazır hale getirir.

Düzeltilen sorunlar:
1. tasks.jsonl oluşturma (tasks.parquet'den veya sıfırdan)
2. episodes.jsonl oluşturma (episodes parquet'den veya data'dan hesaplayarak)
3. stats.json'u güncelleyerek görüntü istatistiklerini ekleme
4. info.json'u güncelleyerek eksik alanları ekleme
5. Dosya yapısını LeRobot formatına dönüştürme
"""

import json
import os
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import subprocess

# Dataset yolu
DATASET_ROOT = Path(__file__).resolve().parent / "output"


def rename_files_to_lerobot_format(dataset_root: Path) -> None:
    """Dosya adlarını LeRobot formatına dönüştürür.
    
    file-000.parquet -> episode_000000.parquet
    file-000.mp4 -> episode_000000.mp4
    
    Zaten doğru formatta olan dosyaları atlar.
    """
    print("\n=== Dosya yapısı LeRobot formatına dönüştürülüyor ===")
    
    # Data dosyalarını yeniden adlandır
    data_dir = dataset_root / "data"
    if data_dir.exists():
        for chunk_dir in sorted(data_dir.iterdir()):
            if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                for parquet_file in sorted(chunk_dir.iterdir()):
                    if parquet_file.suffix == ".parquet" and parquet_file.name.startswith("file-"):
                        file_idx = int(parquet_file.stem.split("-")[1])
                        new_name = f"episode_{file_idx:06d}.parquet"
                        new_path = chunk_dir / new_name
                        if not new_path.exists():
                            shutil.move(str(parquet_file), str(new_path))
                            print(f"  {parquet_file.name} -> {new_name}")
    
    # Video dosyalarını kontrol et ve gerekirse dönüştür
    videos_dir = dataset_root / "videos"
    if videos_dir.exists():
        # Önce mevcut yapıyı kontrol et
        # Doğru yapı: videos/chunk-000/observation.images.rgb/episode_000000.mp4
        # Eski yapı: videos/observation.images.rgb/chunk-000/file-000.mp4
        
        old_format_dirs = []  # Silinecek eski yapıdaki dizinler
        
        for entry in sorted(videos_dir.iterdir()):
            if entry.is_dir():
                # Eğer chunk-XXX formatındaysa, bu zaten yeni yapı
                if entry.name.startswith("chunk-"):
                    continue
                
                # Bu eski yapı olabilir (observation.images.rgb gibi)
                video_key = entry.name
                has_old_format_files = False
                
                for chunk_dir in sorted(entry.iterdir()):
                    if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
                        chunk_name = chunk_dir.name
                        
                        for video_file in sorted(chunk_dir.iterdir()):
                            # Sadece file- formatındaki dosyaları dönüştür
                            if video_file.suffix == ".mp4" and video_file.name.startswith("file-"):
                                has_old_format_files = True
                                file_idx = int(video_file.stem.split("-")[1])
                                new_name = f"episode_{file_idx:06d}.mp4"
                                
                                # Yeni dizin yapısı
                                new_chunk_dir = videos_dir / chunk_name / video_key
                                new_chunk_dir.mkdir(parents=True, exist_ok=True)
                                
                                new_path = new_chunk_dir / new_name
                                if not new_path.exists():
                                    shutil.copy2(str(video_file), str(new_path))
                                    print(f"  {video_key}/{chunk_name}/{video_file.name} -> {chunk_name}/{video_key}/{new_name}")
                
                # Sadece eski format dosyaları varsa bu dizini temizlenecekler listesine ekle
                if has_old_format_files:
                    old_format_dirs.append(entry)
        
        # Eski format dizinlerini temizle
        for old_dir in old_format_dirs:
            try:
                shutil.rmtree(str(old_dir))
                print(f"  Eski yapı temizlendi: {old_dir.name}/")
            except Exception as e:
                print(f"  Uyarı: {old_dir.name}/ temizlenemedi: {e}")
    
    print("  ✓ Dosya yapısı dönüştürüldü")

def load_video_frames_sample(video_path: Path, sample_size: int = 100) -> np.ndarray:
    """Video'dan örnek frame'ler yükler."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Uniform sampling
    if total_frames <= sample_size:
        indices = range(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, sample_size, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.array(frames) if frames else None


def compute_image_stats(dataset_root: Path, video_key: str = "observation.images.rgb", 
                         sample_videos: int = 10, sample_frames_per_video: int = 50) -> dict:
    """Video dosyalarından görüntü istatistiklerini hesaplar."""
    print(f"Görüntü istatistikleri hesaplanıyor: {video_key}")
    
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        print(f"  Video dizini bulunamadı: {videos_dir}")
        return None
    
    # Video dosyalarını bul
    video_files = []
    for chunk_dir in sorted(videos_dir.iterdir()):
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk-"):
            key_dir = chunk_dir / video_key
            if key_dir.exists():
                for video_file in sorted(key_dir.iterdir()):
                    if video_file.suffix == ".mp4":
                        video_files.append(video_file)
    
    if not video_files:
        print(f"  Video dosyası bulunamadı")
        return None
    
    # Örnekleme yap
    if len(video_files) > sample_videos:
        sample_indices = np.linspace(0, len(video_files) - 1, sample_videos, dtype=int)
        video_files = [video_files[i] for i in sample_indices]
    
    print(f"  {len(video_files)} video dosyasından örnekleme yapılıyor...")
    
    all_pixels = []
    for video_file in tqdm(video_files, desc="  Video işleniyor"):
        frames = load_video_frames_sample(video_file, sample_frames_per_video)
        if frames is not None:
            # Normalize to [0, 1]
            frames = frames.astype(np.float32) / 255.0
            # Flatten spatial dimensions, keep channels
            # Shape: (N, H, W, C) -> (N*H*W, C)
            pixels = frames.reshape(-1, frames.shape[-1])
            # Subsample pixels
            if len(pixels) > 10000:
                idx = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[idx]
            all_pixels.append(pixels)
    
    if not all_pixels:
        print(f"  Pixel verisi alınamadı")
        return None
    
    all_pixels = np.concatenate(all_pixels, axis=0)
    
    # İstatistikleri hesapla (per channel)
    mean = all_pixels.mean(axis=0).tolist()
    std = all_pixels.std(axis=0).tolist()
    min_val = all_pixels.min(axis=0).tolist()
    max_val = all_pixels.max(axis=0).tolist()
    
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val
    }


def create_tasks_jsonl(dataset_root: Path) -> None:
    """tasks.jsonl dosyasını oluşturur."""
    print("\n=== tasks.jsonl oluşturuluyor ===")
    
    tasks_jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    tasks_parquet_path = dataset_root / "meta" / "tasks.parquet"
    
    tasks = []
    
    # Eğer tasks.parquet varsa, ondan oku
    if tasks_parquet_path.exists():
        print(f"  tasks.parquet okunuyor...")
        df = pd.read_parquet(tasks_parquet_path)
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Data:\n{df}")
        
        # Sütun adlarını kontrol et
        if 'task_index' in df.columns:
            for idx, row in df.iterrows():
                task_name = df.columns[0] if len(df.columns) > 0 else "robot_arm_control"
                # İlk sütun task adı olabilir
                if isinstance(idx, str):
                    task_name = idx
                tasks.append({
                    "task_index": int(row.get('task_index', 0)) if 'task_index' in df.columns else 0,
                    "task": task_name
                })
        else:
            # Basit format
            for idx, row in df.iterrows():
                tasks.append({
                    "task_index": 0,
                    "task": str(idx) if isinstance(idx, str) else "robot_arm_control"
                })
    
    # Eğer task bulunamadıysa, varsayılan oluştur
    if not tasks:
        print(f"  Varsayılan task oluşturuluyor...")
        tasks = [{"task_index": 0, "task": "robot_arm_control"}]
    
    # JSONL olarak yaz
    with open(tasks_jsonl_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"  ✓ {tasks_jsonl_path} oluşturuldu ({len(tasks)} task)")


def create_episodes_jsonl(dataset_root: Path) -> None:
    """episodes.jsonl dosyasını oluşturur."""
    print("\n=== episodes.jsonl oluşturuluyor ===")
    
    episodes_jsonl_path = dataset_root / "meta" / "episodes.jsonl"
    episodes_parquet_dir = dataset_root / "meta" / "episodes"
    data_dir = dataset_root / "data"
    
    episodes = []
    
    # Episodes parquet'den oku
    if episodes_parquet_dir.exists():
        print(f"  Episodes parquet dosyaları okunuyor...")
        for chunk_dir in sorted(episodes_parquet_dir.iterdir()):
            if chunk_dir.is_dir():
                for parquet_file in sorted(chunk_dir.iterdir()):
                    if parquet_file.suffix == ".parquet":
                        df = pd.read_parquet(parquet_file)
                        for _, row in df.iterrows():
                            # Handle tasks field - convert numpy array to list if needed
                            tasks_val = row["tasks"]
                            if hasattr(tasks_val, 'tolist'):
                                tasks_val = tasks_val.tolist()
                            elif not isinstance(tasks_val, list):
                                tasks_val = [tasks_val]
                            
                            ep_data = {
                                "episode_index": int(row["episode_index"]),
                                "tasks": tasks_val,
                                "length": int(row["length"])
                            }
                            episodes.append(ep_data)
    
    # Eğer episodes bulunamadıysa, data dosyalarından hesapla
    if not episodes:
        print(f"  Data dosyalarından episode bilgileri hesaplanıyor...")
        for chunk_dir in sorted(data_dir.iterdir()):
            if chunk_dir.is_dir():
                for parquet_file in sorted(chunk_dir.iterdir()):
                    if parquet_file.suffix == ".parquet":
                        df = pd.read_parquet(parquet_file)
                        if 'episode_index' in df.columns:
                            ep_idx = df['episode_index'].iloc[0]
                            length = len(df)
                            episodes.append({
                                "episode_index": int(ep_idx),
                                "tasks": ["robot_arm_control"],
                                "length": length
                            })
    
    # Sort by episode_index
    episodes = sorted(episodes, key=lambda x: x["episode_index"])
    
    # JSONL olarak yaz
    with open(episodes_jsonl_path, 'w') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + '\n')
    
    print(f"  ✓ {episodes_jsonl_path} oluşturuldu ({len(episodes)} episode)")


def compute_episodes_stats(dataset_root: Path) -> None:
    """episodes_stats.jsonl dosyasını oluşturur."""
    print("\n=== episodes_stats.jsonl oluşturuluyor ===")
    
    episodes_stats_path = dataset_root / "meta" / "episodes_stats.jsonl"
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    
    def clean_stats(arr):
        """NaN ve Inf değerlerini temizle"""
        result = []
        for val in arr:
            if np.isnan(val) or np.isinf(val):
                result.append(0.0)  # NaN/Inf yerine 0 kullan
            else:
                result.append(float(val))
        return result
    
    def compute_video_stats(video_path: Path, sample_frames: int = 50) -> dict:
        """Video dosyasından görüntü istatistiklerini hesaplar."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames uniformly
        if total_frames <= sample_frames:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        pixels = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB and normalize to [0, 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                # Sample pixels (H, W, C) -> subsample
                h, w, c = frame.shape
                # Take every 10th pixel to reduce computation
                subsampled = frame[::10, ::10, :].reshape(-1, 3)
                pixels.append(subsampled)
        
        cap.release()
        
        if not pixels:
            return None
        
        all_pixels = np.concatenate(pixels, axis=0)
        
        # Compute per-channel stats in format (C, 1, 1) for images
        mean = all_pixels.mean(axis=0)  # (3,)
        std = all_pixels.std(axis=0)    # (3,)
        min_val = all_pixels.min(axis=0)  # (3,)
        max_val = all_pixels.max(axis=0)  # (3,)
        
        # Reshape to (C, 1, 1) format - represented as nested lists [[[val]]] for each channel
        return {
            "mean": [[[float(mean[i])]] for i in range(3)],  # [[[r]], [[g]], [[b]]] -> (3, 1, 1)
            "std": [[[float(std[i])]] for i in range(3)],
            "min": [[[float(min_val[i])]] for i in range(3)],
            "max": [[[float(max_val[i])]] for i in range(3)],
            "count": [total_frames]
        }
    
    # Her episode için - hem eski (file-xxx) hem yeni (episode_xxx) formatı destekle
    episode_files = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir():
            for parquet_file in sorted(chunk_dir.iterdir()):
                if parquet_file.suffix == ".parquet":
                    # Hem file-xxx hem episode_xxx formatını kabul et
                    if parquet_file.name.startswith("file-") or parquet_file.name.startswith("episode_"):
                        episode_files.append(parquet_file)
    
    episodes_stats = []
    for ep_idx, parquet_file in enumerate(tqdm(episode_files, desc="  Episode stats")):
        df = pd.read_parquet(parquet_file)
        
        stats = {}
        num_frames = len(df)
        
        # observation.state stats
        if 'observation.state' in df.columns:
            states = np.stack(df['observation.state'].values)
            stats['observation.state'] = {
                "mean": clean_stats(states.mean(axis=0)),
                "std": clean_stats(states.std(axis=0)),
                "min": clean_stats(states.min(axis=0)),
                "max": clean_stats(states.max(axis=0)),
                "count": [num_frames]  # Must be a list with single element to match (1,) shape
            }
        
        # action stats
        if 'action' in df.columns:
            actions = np.stack(df['action'].values)
            stats['action'] = {
                "mean": clean_stats(actions.mean(axis=0)),
                "std": clean_stats(actions.std(axis=0)),
                "min": clean_stats(actions.min(axis=0)),
                "max": clean_stats(actions.max(axis=0)),
                "count": [num_frames]  # Must be a list with single element to match (1,) shape
            }
        
        # observation.images.rgb stats - video dosyasından hesapla
        # Video path: videos/chunk-{chunk}/observation.images.rgb/episode_{idx}.mp4
        chunk_name = parquet_file.parent.name  # e.g., "chunk-000"
        video_key = "observation.images.rgb"
        video_path = videos_dir / chunk_name / video_key / f"episode_{ep_idx:06d}.mp4"
        
        if video_path.exists():
            video_stats = compute_video_stats(video_path)
            if video_stats:
                stats['observation.images.rgb'] = video_stats
        
        ep_stats = {
            "episode_index": ep_idx,
            "stats": stats
        }
        episodes_stats.append(ep_stats)
    
    # JSONL olarak yaz
    with open(episodes_stats_path, 'w') as f:
        for ep_stat in episodes_stats:
            f.write(json.dumps(ep_stat) + '\n')
    
    print(f"  ✓ {episodes_stats_path} oluşturuldu ({len(episodes_stats)} episode)")


def update_stats_json(dataset_root: Path) -> None:
    """stats.json dosyasını görüntü istatistikleri ile günceller."""
    print("\n=== stats.json güncelleniyor ===")
    
    stats_path = dataset_root / "meta" / "stats.json"
    
    # Mevcut stats'ı yükle
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    print(f"  Mevcut stats anahtarları: {list(stats.keys())}")
    
    # info.json'dan features'ı oku
    info_path = dataset_root / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    features = info.get("features", {})
    
    # Görüntü özellikleri için istatistik hesapla
    for feature_name, feature_info in features.items():
        if feature_info.get("dtype") == "video" and feature_name not in stats:
            print(f"\n  {feature_name} için istatistikler hesaplanıyor...")
            image_stats = compute_image_stats(dataset_root, feature_name)
            if image_stats:
                stats[feature_name] = image_stats
    
    # Mevcut stats'ı kontrol et ve eksik olanları hesapla
    data_dir = dataset_root / "data"
    
    # observation.state ve action için istatistikleri yeniden hesapla
    all_states = []
    all_actions = []
    
    print("\n  Tüm data dosyalarından istatistikler hesaplanıyor...")
    for chunk_dir in sorted(data_dir.iterdir()):
        if chunk_dir.is_dir():
            for parquet_file in tqdm(list(chunk_dir.iterdir()), desc=f"  {chunk_dir.name}"):
                if parquet_file.suffix == ".parquet":
                    df = pd.read_parquet(parquet_file)
                    if 'observation.state' in df.columns:
                        all_states.extend(df['observation.state'].tolist())
                    if 'action' in df.columns:
                        all_actions.extend(df['action'].tolist())
    
    if all_states:
        states = np.array(all_states)
        stats['observation.state'] = {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist()
        }
        print(f"  ✓ observation.state stats güncellendi")
    
    if all_actions:
        actions = np.array(all_actions)
        stats['action'] = {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist()
        }
        print(f"  ✓ action stats güncellendi")
    
    # Stats'ı kaydet
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n  ✓ {stats_path} güncellendi")
    print(f"  Final stats anahtarları: {list(stats.keys())}")


def update_info_json(dataset_root: Path) -> None:
    """info.json dosyasını günceller."""
    print("\n=== info.json güncelleniyor ===")
    
    info_path = dataset_root / "meta" / "info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # chunks_size ekle (eğer yoksa)
    if "chunks_size" not in info:
        info["chunks_size"] = 1000
        print("  ✓ chunks_size eklendi")
    
    # data_path formatını güncelle - LeRobot beklediği format
    old_data_path = info.get("data_path", "")
    expected_data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if "chunk_index" in old_data_path or "file_index" in old_data_path:
        info["data_path"] = expected_data_path
        print(f"  ✓ data_path güncellendi: {expected_data_path}")
    
    # video_path formatını güncelle - LeRobot beklediği format
    old_video_path = info.get("video_path", "")
    expected_video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    if "chunk_index" in old_video_path or "file_index" in old_video_path:
        info["video_path"] = expected_video_path
        print(f"  ✓ video_path güncellendi: {expected_video_path}")
    
    # codebase_version kontrolü
    if info.get("codebase_version") not in ["v2.0", "v2.1", "v3.0"]:
        info["codebase_version"] = "v2.1"
        print(f"  ✓ codebase_version güncellendi: v2.1")
    
    # Kaydet
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  ✓ {info_path} güncellendi")


def validate_dataset(dataset_root: Path) -> bool:
    """Dataset'in doğru formatta olduğunu kontrol eder."""
    print("\n=== Dataset Doğrulama ===")
    
    required_files = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = dataset_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - EKSIK!")
            all_ok = False
    
    # Data ve video dizinlerini kontrol et
    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    
    if data_dir.exists():
        num_parquets = len(list(data_dir.rglob("*.parquet")))
        print(f"  ✓ data/ ({num_parquets} parquet dosyası)")
    else:
        print(f"  ✗ data/ - EKSIK!")
        all_ok = False
    
    if videos_dir.exists():
        num_videos = len(list(videos_dir.rglob("*.mp4")))
        print(f"  ✓ videos/ ({num_videos} video dosyası)")
    else:
        print(f"  ✗ videos/ - EKSIK!")
        all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("LeRobot Dataset Metadata Fixer")
    print("=" * 60)
    print(f"\nDataset: {DATASET_ROOT}")
    
    if not DATASET_ROOT.exists():
        print(f"HATA: Dataset dizini bulunamadı: {DATASET_ROOT}")
        sys.exit(1)
    
    # 0. Dosya yapısını LeRobot formatına dönüştür
    rename_files_to_lerobot_format(DATASET_ROOT)
    
    # 1. tasks.jsonl oluştur
    create_tasks_jsonl(DATASET_ROOT)
    
    # 2. episodes.jsonl oluştur
    create_episodes_jsonl(DATASET_ROOT)
    
    # 3. episodes_stats.jsonl oluştur
    compute_episodes_stats(DATASET_ROOT)
    
    # 4. info.json güncelle
    update_info_json(DATASET_ROOT)
    
    # 5. stats.json güncelle (görüntü istatistikleri dahil)
    update_stats_json(DATASET_ROOT)
    
    # 6. Doğrulama
    print("\n" + "=" * 60)
    if validate_dataset(DATASET_ROOT):
        print("\n✓ Dataset başarıyla düzeltildi ve eğitime hazır!")
    else:
        print("\n✗ Bazı dosyalar eksik. Lütfen kontrol edin.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
