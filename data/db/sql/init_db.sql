CREATE TABLE reddit_posts(
    archived boolean,
    author TEXT,
    author_flair_css_class TEXT,
    author_flair_text TEXT,
    brand_safe boolean,
    contest_mode boolean,
    created_utc bigint,
    distinguished TEXT,
    domain TEXT,
    edited bigint,
    gilded integer,
    hidden boolean,
    hide_score boolean,
    id TEXT,
    is_self boolean,
    link_flair_css_class TEXT,
    link_flair_text TEXT,
    locked boolean,
    media TEXT,
    media_embed TEXT,
    num_comments INTEGER,
    over_18 boolean,
    permalink TEXT,
    post_hint TEXT,
    preview TEXT,
    quarantine boolean,
    retrieved_on bigint,
    score INTEGER,
    secure_media TEXT,
    secure_media_embed TEXT,
    selftext TEXT,
    spoiler boolean,
    stickied boolean,
    subreddit TEXT,
    subreddit_id TEXT,
    suggested_sort TEXT,
    thumbnail TEXT,
    title TEXT,
    url TEXT,
    hashed_author TEXT,
    upvote_ratio double precision,
    qLevel integer
);

CREATE TABLE qanon_authors(
    q_author TEXT,
    is_uq boolean,
    status TEXT
);

COPY qanon_authors from '../../QAnon/Hashed_allAuthorStatus.csv'
DELIMITER ','
CSV HEADER;