COPY (
select
p.hashed_author,
p.score,
p.num_comments,
p.title,
p.selftext,
(p.q_level/(p.q_level+0.1))::INTEGER as q_level, 
p.created_utc
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
order by hashed_author, created_utc
) 
TO STDOUT 
WITH (FORMAT CSV, HEADER);
