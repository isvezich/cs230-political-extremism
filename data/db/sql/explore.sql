select 12;

CREATE TABLE reddit_posts_processed(
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
    status TEXT,
    q_level integer
);

CREATE TABLE qanon_authors(
    q_author TEXT,
    is_uq boolean,
    status TEXT
);

select * from qanon_authors
limit 10;

COPY qanon_authors from '../Downloads/Hashed_allAuthorStatus.csv'
DELIMITER ','
CSV HEADER;

drop table reddit_posts_processed;

INSERT into reddit_posts_processed
SELECT
	archived,
    author,
    author_flair_css_class,
    author_flair_text,
    brand_safe,
    contest_mode,
    created_utc,
    distinguished,
    domain,
    edited,
    gilded,
    hidden,
    hide_score,
    id,
    is_self,
    link_flair_css_class,
    link_flair_text,
    locked,
    media,
    media_embed,
    num_comments,
    over_18,
    permalink,
    post_hint,
    preview,
    quarantine,
    retrieved_on,
    score,
    secure_media,
    secure_media_embed,
    selftext,
    spoiler,
    stickied,
    subreddit,
    subreddit_id,
    suggested_sort,
    thumbnail,
    title,
    url,
    hashed_author,
    upvote_ratio,
    status,
case when is_uq is null then 0 when is_uq then 2 else 1 end as q_level
  FROM reddit_posts as r
  LEFT JOIN qanon_authors as q
    ON r.hashed_author = q.q_author;


select * from reddit_posts_processed limit 10;

SELECT selftext, title, hashed_author, subreddit FROM reddit_posts_processed limit 30;

CREATE INDEX IF NOT EXISTS idx_reddit_posts_subreddit ON reddit_posts (subreddit);

CREATE INDEX IF NOT EXISTS idx_reddit_posts_hashed_author ON reddit_posts (hashed_author);


create view posts_good as
select *
from reddit_posts_processed
where (score > 1 or num_comments > 0)

-- Count authors / posts

--all authors
select count(1) from (select distinct hashed_author from reddit_posts_processed) as t;
-- 1,967,163

--all authors with "good" posts
select count(1) from (select distinct hashed_author from reddit_posts_processed 
where (score > 1 or num_comments > 0)) as t;
-- 1,706,062


--q authors
select count(1) from (select distinct hashed_author from reddit_posts_processed where q_level <> 0) as t;
-- 13182

--q authors with "good" posts
select count(1) from
(select distinct hashed_author from posts_good
where q_level <> 0) as t;
-- 12,818

--q authors with "good" posts made before 2017-10-01
select count(1) from
(select distinct hashed_author from posts_good
where q_level <> 0 and created_utc < '2017-10-01') as t;


select extract(epoch from '2017-10-01'::TIMESTAMP);


-- num posts
select count(*) from reddit_posts_processed;
-- 11,318,388


-- Sample positive examples (q authors)
select hashed_author,
score,
num_comments,
title,
selftext,
q_level , 
created_utc
from posts_good
where q_level <> 0
order by hashed_author, created_utc;


COPY (select hashed_author,
score,
num_comments,
title,
selftext,
q_level , 
created_utc
from posts_good
where q_level <> 0
order by hashed_author, created_utc
) 
TO '/C/DATA/q-author-posts.csv' 
WITH (FORMAT CSV, HEADER);

select (2/(2+0.1))::INTEGER;

-- Sample negative examples (non q authors)

select 
hashed_author,
score,
num_comments,
title,
selftext,
q_level , 
created_utc
from posts_good
where hashed_author in
(
select hashed_author
from posts_good
where q_level = 0 
group by hashed_author
having count(*) > 10 and count(*) < 100
order by random()
limit 12818
)
order by hashed_author, created_utc;


-- Filter out spammy posts/authors

select log_score, count(*) 
from (
select hashed_author, log(2, score)::integer, count(*)
from reddit_posts_processed
where q_level <> 0 and abs(score) > 1
group by log(2, score)::integer, hashed_author
order by log(2, score)::integer desc, count(*) desc
) as x(auth, log_score, n)
group by log_score
order by log_score desc;


select hashed_author, count(*)
from reddit_posts_processed
where q_level <> 0 and (status = 'Active') and score > 1
group by hashed_author
order by count(*) desc;

select * from reddit_posts_processed where hashed_author = '00159fd6226c487f6898f61533897e87d2f91d54' limit 100;

select distinct status from reddit_posts_processed where q_level <> 0

select * from  pg_type;