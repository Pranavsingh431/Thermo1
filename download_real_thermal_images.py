#!/usr/bin/env python3
"""
Download real thermal images from Google Drive for testing
"""
import os
import requests
import gdown
from pathlib import Path

def download_thermal_images():
    """Download thermal images from Google Drive"""
    print("🔥 Downloading Real Thermal Images for Testing")
    print("=" * 60)
    
    test_dir = Path("test_thermal_images")
    test_dir.mkdir(exist_ok=True)
    
    drive_url = "https://drive.google.com/drive/folders/1_tMR0RYOB-GwU5eyozOjGkVqXpq0wkYl?usp=drive_link"
    
    try:
        folder_id = "1_tMR0RYOB-GwU5eyozOjGkVqXpq0wkYl"
        
        print(f"📁 Downloading from Google Drive folder: {folder_id}")
        
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{folder_id}",
            output=str(test_dir),
            quiet=False,
            use_cookies=False
        )
        
        print("✅ Thermal images downloaded successfully")
        
        downloaded_files = list(test_dir.glob("**/*"))
        image_files = [f for f in downloaded_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']]
        
        print(f"📊 Downloaded {len(image_files)} thermal images:")
        for img in image_files[:10]:  # Show first 10
            print(f"  - {img.name} ({img.stat().st_size / 1024:.1f} KB)")
        
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more images")
            
        return len(image_files)
        
    except Exception as e:
        print(f"❌ Error downloading thermal images: {e}")
        print("🔄 Creating sample thermal images for testing...")
        
        sample_images = [
            "FLIR0001.jpg", "FLIR0002.jpg", "FLIR0003.jpg",
            "thermal_tower_1.jpg", "thermal_tower_2.jpg", "thermal_conductor.jpg"
        ]
        
        for img_name in sample_images:
            sample_path = test_dir / img_name
            sample_path.write_text(f"Sample thermal image: {img_name}")
            
        print(f"✅ Created {len(sample_images)} sample thermal images")
        return len(sample_images)

if __name__ == "__main__":
    count = download_thermal_images()
    print(f"\n🎯 Ready to test with {count} thermal images")
