# The largest sum of a non-empty run, on an array of numbers.
# A run is the difference of two running sums, so the best run that ends at a value is its running sum minus the smallest running sum before it.
# The running sums have a zero in front, for the runs that begin at the start.
# The running sums after the first, minus the running smallest of the ones before each, and the largest of those.
MaxSub ← /↥ - ⊃(↘¯1\↧|↘1) ⊂0\+
