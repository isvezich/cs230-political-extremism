import json
import csv
import hashlib
import sys

keys = ["archived","author","author_flair_css_class","author_flair_text","brand_safe","contest_mode","created_utc",
        "distinguished","domain","edited","gilded","hidden","hide_score","id","is_self","link_flair_css_class",
        "link_flair_text","locked","media","media_embed","num_comments","over_18","permalink","post_hint","preview",
        "quarantine","retrieved_on","score","secure_media","secure_media_embed","selftext","spoiler","stickied",
        "subreddit","subreddit_id","suggested_sort","thumbnail","title","url", "hashed_author", "upvote_ratio", "qLevel"]


def sha1(x):
    m = hashlib.sha1()
    m.update(x.encode())
    m.digest()
    return m.hexdigest()


writer = csv.writer(sys.stdout, quoting=csv.QUOTE_MINIMAL)
for line in sys.stdin:
    data = json.loads(line)
    row = []
    for key in keys:
        if data.get(key) is not None:
            if (key == 'edited' or key == 'gilded') and data[key] is False:
                row.append(None)
            else:
                row.append(data[key])
        elif key == 'qLevel':
            # Set default qLevel of non qAnon posts to 0
            row.append(0)
        elif key == 'hashed_author':
            row.append(sha1(data['author']))
        else:
            row.append(None)
    writer.writerow(row)
