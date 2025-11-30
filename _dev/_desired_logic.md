
# Deisred Database Logic

CaliResults ID: None  Experiment ID:None  Detection ID: None  Extraction ID: None  Analysis ID: None
Pos: []

Experiment ID: 1

Run Detection Only:
    - on pos [0]
    - detection settings id: 1
    - CaliResults ID: 1  Experiment ID: 1  Detection ID: 1  Extraction ID: None  Analysis ID: None
      Pos: [0]

Run Detection Only:
    - on pos [0, 1]
    - detection settings id: 1
    - we skip detection on pos [0] since already exists
    - CaliResults ID: 1  Experiment ID: 1  Detection ID: 1  Extraction ID: None  Analysis ID: None
      Pos: [0, 1]

Run Detection + Extraction:
    - on pos [0, 2]
    - detection settings id: 1
    - extraction settings id: 1
    - we skip detection on pos [0] since already exists and only run on pos [2]
    - we run extraction on pos [0, 2]
    - CaliResults ID: 1  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: None
      Pos: [0, 1, 2]

Run Analysis Only:
    - on pos [0, 3]
    - detection settings id: 1
    - extraction settings id: 1
    - analysis settings id: 1
    - we skip detection on pos [0] since already exists and only run on pos [3]
    - we skip extraction on pos [0] since already exists and only run on pos [3]
    - we run analysis on pos [0, 3]
    - CaliResults ID: 1  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: 1
      Pos: [0, 1, 2, 3]

Run Detection + Extraction + Analysis:
    - on pos [0, 1, 4]
    - detection settings id: 1
    - extraction settings id: 1
    - analysis settings id: 1
    - we skip detection on pos [0, 1] since already exists and only run on pos [4]
    - we skip extraction on pos [0] since already exists and only run on pos [1, 4]
    - we skip analysis on pos [0] since already exists and only run on pos [1, 4]
    - CaliResults ID: 1  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: 1
      Pos: [0, 1, 2, 3, 4]

Run Detection + Extraction + Analysis:
    - on pos [0, 5]
    - detection settings id: 2
    - extraction settings id: 1
    - analysis settings id: 1
    - we run detection on pos [0, 5] since detection settings id: 2 is new
    - we run extraction on pos [0, 5] with the same extraction settings id: 1
    - we run analysis on pos [0, 5] with the same analysis settings id
    - CaliResults ID: 2  Experiment ID: 1  Detection ID: 2  Extraction ID: 1  Analysis ID: 1
      Pos: [0, 5]

Run Detection + Extraction + Analysis:
    - on pos [0, 1, 4]
    - detection settings id: 1
    - extraction settings id: 1
    - analysis settings id: 2
    - we skip detection on pos [0, 1, 4] since already exists
    - we skip extraction on pos [0, 1, 4] since already exists
    - we run analysis on pos [0, 1, 4] with the new analysis settings id: 2
    - CaliResults ID: 3  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: 2
      Pos: [0, 1, 4]

Run Detection + Extraction + Analysis:
    - on pos [5, 6, 7]
    - detection settings id: 1
    - extraction settings id: 1
    - analysis settings id: 2
    - we run detection on pos [5, 6, 7] with the same detection settings id: 1
    - we run extraction on pos [5, 6, 7] with the same extraction settings id: 1
    - we run analysis on pos [5, 6, 7] with the same analysis settings id: 2
    - however, let's say we cancel the run after pos 6 detection
        - pos [5] is detected but no extracted or analyzed
        - pos [6, 7] is not detected, extracted, or analyzed
    - CaliResults ID: 3  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: 2
      Pos: [0, 1, 4, 5]

Run Detection + Extraction + Analysis:
    - on pos [5, 6, 7]
    - detection settings id: 1
    - extraction settings id: 1
    - analysis settings id: 2
    - we skip detection on pos since already exists [5] but run on pos [6, 7] with detection settings id: 1
    - we run extraction on pos [5, 6, 7] with the same extraction settings id: 1
    - we run analysis on pos [5, 6, 7] with the same analysis settings id: 2
    - however, let's say we cancel the run after extraction of pos 6
        - pos [5] was already detected and is now extracted and analyzed
        - pos [6] is detected and extracted but not analyzed
        - pos [7] is detected but not extracted or analyzed
    - CaliResults ID: 3  Experiment ID: 1  Detection ID: 1  Extraction ID: 1  Analysis ID: 2
      Pos: [0, 1, 4, 5, 6, 7]
