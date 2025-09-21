Smart File Type Prediction
Problem:
Just checking a file’s name or extension isn’t safe, because someone can rename a harmful file (like a program .exe) to look like a picture .jpg.

ML Solution
The model looks inside the file’s raw data (its first bytes), learns the hidden patterns of each file type, and predicts what the file really is. If that doesn’t match the given extension, it’s flagged as suspicious.
