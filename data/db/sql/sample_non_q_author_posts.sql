copy (
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
order by hashed_author, created_utc
)
TO STDOUT
WITH (FORMAT CSV, HEADER);
