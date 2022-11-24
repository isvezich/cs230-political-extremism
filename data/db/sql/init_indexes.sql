CREATE INDEX IF NOT EXISTS idx_reddit_posts_subreddit ON reddit_posts (subreddit);
CREATE INDEX IF NOT EXISTS idx_reddit_posts_hashed_author ON reddit_posts (hashed_author);
