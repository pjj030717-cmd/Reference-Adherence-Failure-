{
  "system": "Choose the option that is factually correct for the question.\nReply with only A or B.",
  "user_template": "Question:\n{q}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nAnswer:",
  "note": "no Reference / Candidate / Judge wording; ordinary two-choice factual QA",
  "continuation": {
    "A": " A",
    "A_id": 362,
    "B": " B",
    "B_id": 425
  },
  "teacher_forcing_pos": "prompt_len - 1",
  "k_definition": "k = (d_1 + d_2) / 2  with d_1 = l_A - l_B (Order1: A=r_o,B=r_s), d_2 = l_B - l_A (Order2: A=r_s,B=r_o)"
}