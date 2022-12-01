COPY (
select hashed_author,
score,
num_comments,
title,
selftext,
(q_level/(q_level+0.1))::INTEGER , 
created_utc
from posts_good
where q_level <> 0 and random() < 0.25
order by hashed_author, created_utc
) 
TO STDOUT 
WITH (FORMAT CSV, HEADER);
