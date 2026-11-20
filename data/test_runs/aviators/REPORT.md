# Aviators resumes: pipeline test

Job description: FY'26 NCG Full Time SWE JD_Visa.pdf
Screening id: `a3c12c12-6db0-4d4a-8466-2e852f9fc3d2`

- Uploaded/indexed: 25 resumes, 180 chunks; maximum 382 tokens per chunk.
- Shortlisted: 10; evaluated: 10; errors: 0; filtered: 0.
- Evidence quotes checked against anonymized stored resume chunks: 29.
- Unverified evidence flags: 14; these quotes were removed and confidence reduced. Template summary fallbacks: 0.
- Stored scores match the code scoring formula. All candidates belong to the selected folder.

## Parsed requirements

- Any of: Proficiency in at least one programming language or technology including, but not limited, to Java, C/C++, C#, .NET, python, Javascript, HTML, CSS and Objective-C
- The ability to take on challenges and address problems head-on
- Highly driven and resourceful
- Strong ability to collaborate

Degree/graduation eligibility remains in the job context for fit assessment. The language alternatives are one requirement; no preferred skills were invented.

## Ranked results

| Rank | Resume | Score | Recommendation | Flags |
|---|---|---:|---|---:|
| 1 | AnujYadav-3AugIIITM.pdf | 91.75 | interview | 0 |
| 2 | Jayanth_Resume_mondee.pdf | 67.50 | maybe | 0 |
| 3 | VishalDaware_Resume.pdf | 55.81 | maybe | 1 |
| 4 | Atharva_Arbat_AI_v3.pdf | 47.31 | not_a_fit | 1 |
| 5 | T Mohan Lakshmi Ganesh_Resume.pdf | 44.12 | not_a_fit | 2 |
| 6 | adarsh_res6-1.pdf | 43.44 | not_a_fit | 3 |
| 7 | Sabrin_S_Resume.pdf | 42.25 | not_a_fit | 0 |
| 8 | Daiwik_s Saxena Resume.pdf | 37.80 | not_a_fit | 0 |
| 9 | Nikshith.pdf | 23.94 | not_a_fit | 3 |
| 10 | Mageshwari M.pdf | 20.75 | not_a_fit | 4 |

## Findings and limits

- Fixed test scoping so synthetic calibration resumes cannot enter this run.
- The initial LLM extraction mixed degree eligibility into skill proficiency and inferred preferences from possible work areas. A regression-tested normalization step corrects these cases; the first attempt is retained separately in the manifest.
- Constrained per-position skill names prevent omission or duplication of grouped requirements. Invalid numeric scales are rejected and retried with field-specific feedback, without logging resume text.
- The configured shortlist evaluates 10 of the 25 indexed resumes. The other 15 received no detailed LLM evaluation in this test.
- No human expected ranking was supplied; execution, evidence grounding, and score arithmetic were tested, while ranking accuracy and graduation eligibility require human review.
- This tool recommends. A human makes the final decision.
