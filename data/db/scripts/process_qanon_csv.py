import csv
import sys
from datetime import datetime, timezone

keys = ["archived","author","author_flair_css_class","author_flair_text","brand_safe","contest_mode","created_utc",
        "distinguished","domain","edited","gilded","hidden","hide_score","id","is_self","link_flair_css_class",
        "link_flair_text","locked","media","media_embed","num_comments","over_18","permalink","post_hint","preview",
        "quarantine","retrieved_on","score","secure_media","secure_media_embed","selftext","spoiler","stickied",
        "subreddit","subreddit_id","suggested_sort","thumbnail","title","url", "hashed_author", "upvote_ratio", "qLevel"]
q_anon_keys = ['subreddit','id','score','numReplies','author','title','text','is_self','domain','url','permalink',
               'upvote_ratio','date_created']
qanon_to_db_column = {
    'numReplies': 'num_comments',
    'text': 'selftext',
    'author': 'hashed_author',
    'date_created': 'created_utc',
}

writer = csv.writer(sys.stdout, quoting=csv.QUOTE_MINIMAL)
with open('../QAnon/Hashed_Q_Submissions_Raw_Combined.csv', 'rU') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
    next(reader)
    for line in reader:
        count = 0
        key_to_value = {}
        for word in line:
            currKey = q_anon_keys[count]
            if qanon_to_db_column.get(currKey):
                if currKey == 'date_created':
                    # Convert to UTC timestamp
                    dt = datetime.strptime(word, "%Y-%m-%d %H:%M:%S")
                    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
                    word = int(timestamp)
                currKey = qanon_to_db_column[currKey]
            key_to_value[currKey] = word
            count += 1

        row = []
        for key in keys:
            if key_to_value.get(key):
                row.append(key_to_value[key])
            else:
                row.append(None)

        writer.writerow(row)
