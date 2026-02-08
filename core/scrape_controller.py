class ScrapeController:
def __init__(self, loader, scraper, storage):
"""
loader: loads dataset into memory
scraper: has method scrape_recent(n) -> list[match_dict]
storage: handles saving/appending to CSV
"""
self.loader = loader
self.scraper = scraper
self.storage = storage

def scrape_and_merge(self, n: int):
df = self.loader.get_df()
known = set(df["match_id"].astype(str)) if "match_id" in df.columns else set()

pulled = self.scraper.scrape_recent(n)  # list of dict rows with match_id
new_rows = [r for r in pulled if str(r.get("match_id")) not in known]

if new_rows:
self.storage.append_rows(new_rows)  # append to CSV
self.loader.reload()                # reload df in memory
return len(new_rows)
