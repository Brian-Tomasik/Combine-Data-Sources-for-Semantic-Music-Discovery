README for code associated with the music information combiner
Brian Tomasik
May 2009

The relevant Python files here are

combiner.py
crossvalidate.py
ranker.py
to_latex.py
util.py

The first four have pydocs that give further details when you run

import combiner
help(combiner)

or similarly for the other three. util.py simply contains utility functions used
by the other code.

There are two main operations one might do with this code:

(1) Generate a combined song-tag score matrix for use by a query ranker.
For this, check that you have no existing files named "final.tab" that can't be
overwritten. If not, then type

python combiner.py

from within the current directory. The new results will appear as "final.tab".
(Note: This only works from within a directory structure of the correct type.
The details of the directory structure and required files can be deciphered
from hard-coded values in combiner.py.)

With this file, one can then use the Ranker module in the manner shown
in the main() function of ranker.py to answer queries.

(2) Run cross-validation experiments. In this case, set the main() function
of crossvalidate.py to run the experiment you'd like (perhaps using
the tune_param function, or else using one of the standard parameter-
tuning runs that have been made into their own public functions). Then
type

python crossvalidate.py

from within the current directory. The results will be written to a timestamped
directory precise to the hour, so unless you've just run this code within the 
past hour, your old results won't be overwritten.

----

Here's a list of to-do items if interested:

- Currently, in combiner.py, the _compute_betas function sets to 0 any
regression coefficients that came out negative, except in the cases where we
deliberately allow negative coefficients (for the popularity source or if we're
doing a regression that includes interactions). There are proper ways to do
non-negative regression. A quick online search turns up some academic
articles and R discussion lists. Since we're doing ordinary independent
regression by default, it may suffice to use a function that does "non-negative
least squares" (search the term to find lots of references).
- Try cross-validation experiments using the last.fm social tags. This would
consist of adding "Last.fm" to the list of default tagtypes_to_run in 
crossvalidate.py's tune_param function. I'm not sure if code breaks when this
is done.
