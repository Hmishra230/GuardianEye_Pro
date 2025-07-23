from event_db import EventDB

db = EventDB()
events = db.search_events()
print(f'Found {len(events)} events')

# Show last 5 events
for event in events[-5:]:
    print(f'Event {event["id"]}: snapshot_path = {event.get("snapshot_path", "None")}')
