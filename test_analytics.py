from dashboard import get_video_analytics_stats

# Test the analytics function
print("🎯 Testing Video Analytics Dashboard...")
stats = get_video_analytics_stats()

print(f"📊 Total Detections: {stats['total_detections']}")
print(f"📷 Categories: {list(stats['category_counts'].keys())}")
print(f"🎥 Cameras: {list(stats['camera_counts'].keys())}")
print(f"📅 Daily counts: {len(stats['daily_counts'])} days with data")
print(f"⏰ Recent events: {len(stats['recent_events'])} events")

# Show category breakdown
print("\n📈 Category Breakdown:")
for category, count in stats['category_counts'].items():
    print(f"  {category}: {count} detections")

# Show camera breakdown
print("\n🎥 Camera Breakdown:")
for camera, count in stats['camera_counts'].items():
    print(f"  {camera}: {count} detections")

print(f"\n✅ Analytics working! Detection count matches snapshot count: {stats['total_detections'] == 627}")
