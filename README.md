# Linguistic Analysis of Scottish School Exam Papers #
## Intent ##
The Scottish School Exam Papers dataset provides a unique opportunity to explore how the language used in education reflects evolving societal, pedagogical, and linguistic trends. By examining the phrasing, structure, and vocabulary of exam questions over time, this project aims to uncover shifts in accessibility, inclusivity, and readability within Scotland's education system. Such an analysis can offer insights into how language in assessments aligns with broader historical, social, and educational changes.

By measuring linguistic features such as sentence length, complexity, and vocabulary diversity, trends in the evolution of exam question readability can be identified. For instance, earlier exams may reveal long, complex sentence structures reflective of 19th-century academic norms, while more modern exams could display shorter, more direct phrasing influenced by contemporary pedagogical principles emphasising clarity and accessibility.

Another dimension of the analysis focuses on inclusivity in language. Historical exams may reveal implicit biases in phrasing or content, such as gendered language or cultural assumptions that privilege specific societal groups. Tracking changes in the language used to describe historical figures, literature, or scientific phenomena can highlight how educational materials have adapted to incorporate broader perspectives and ensure representation.

## Progress ##
* Done
  * Read NLS text files
  * Split into question (1, 2, 3, etc.)
  * Split into subquestions (a, b, c, etc.)
  * Perform "coleman liau", "flesch kincaid", and"gunning fog" analysis on questions and subquestions
  * Output to CSV
