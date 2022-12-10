import psycopg2
import os
import gzip

conn = psycopg2.connect("dbname='reddit' user='postgres' host='localhost' password='asdf'")

out_dir = './mydata_txt'

def make_dirs(q_level, author):
    path = os.path.join(out_dir, str(q_level), author)
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def write_post_txt(text, dir, post_id, ts):
    path = os.path.join(dir, "{}_{}.txt".format(ts, post_id))
    with open(path, 'w') as f:
        f.write(text)

def write_post_gz(text, dir, post_id, ts):
    path = os.path.join(dir, "{}_{}.txt.gz".format(ts, post_id))
    with gzip.open(path, 'wb') as f:
        f.write(text.encode())

def do_pg_q():
    cursor = conn.cursor("x")
    cursor.execute("""
select
p.id, -- 0
p.hashed_author as author, -- 1
concat_ws(' ', p.title, p.selftext),  -- 2
(p.q_level/(p.q_level+0.1))::INTEGER as q_level, -- 3
p.created_utc -- 4
from
posts_good as p
join
(
select hashed_author, min(created_utc) as ts from posts_good
where q_level <> 0 and subreddit in (
'greatawakening',
'WWG1WGA01',
'The_GreatAwakening',
'TheCalmBeforeTheStorm',
'AFTERTHESTQRM',
'BiblicalQ',
'TheGreatAwakening',
'thestorm',
'QAnon',
'QProofs',
'QGreatAwakening',
'QanonandJFKjr',
'QanonUK',
'qresearch',
'QanonTools',
'NorthernAwakening',
'greatawakening2',
'Q4U',
'Quincels'
)
group by hashed_author
) as d
on p.hashed_author = d.hashed_author
where p.created_utc < d.ts
--  and random() < 0.25
order by p.hashed_author, p.created_utc
-- limit 1
""")
    for r in cursor:
        dir = make_dirs(r[3], r[1])
        write_post_txt(r[2], dir, r[0], r[4])


def do_pg_noq():
    cursor = conn.cursor("y")
    cursor.execute("""
select
p.id, -- 0
p.hashed_author as author, -- 1
concat_ws(' ', p.title, p.selftext),  -- 2
(p.q_level/(p.q_level+0.1))::INTEGER as q_level, -- 3
p.created_utc -- 4
from
posts_good as p
where p.hashed_author in
(
select hashed_author
from posts_good
where q_level = 0 
group by hashed_author
having count(*) > 20 and count(*) < 1000
order by random()
limit 6185
)
order by p.hashed_author, created_utc
-- limit 1
""")
    for r in cursor:
        dir = make_dirs(r[3], r[1])
        write_post_txt(r[2], dir, r[0], r[4])



do_pg_q()
do_pg_noq()