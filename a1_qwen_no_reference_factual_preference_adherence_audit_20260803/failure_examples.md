# failure_examples.md

> 本轮 k 与 y_SS 关联的例外样本（按数据集挑取各 3 条）：高 k 却 y_SS=0，低 k 却 y_SS=1。

## SciQ 398a76fc83c2…

- k = 23.555；d_1 = 23.359；d_2 = 23.750；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: What do you call air flowing over earth’s surface?
- r_o: wind
- r_s: concave

## SciQ ec9503c6deb0…

- k = 23.453；d_1 = 22.375；d_2 = 24.531；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: What type of evolution happens when two species evolve the same traits?
- r_o: convergent
- r_s: cloacal bladder

## SciQ d7bf00bba92d…

- k = 23.281；d_1 = 23.344；d_2 = 23.219；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: What is the term for a partial degradation of glucose without the use of oxygen?
- r_o: fermentation
- r_s: myoglobin

## SciQ 1b113718103d…

- k = 7.719；d_1 = 8.250；d_2 = 7.188；order_consistent = True
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: What do most of the noble gas elements have in common?
- r_o: eight valence electrons
- r_s: ice

## SciQ 84ab9cc3d020…

- k = 10.359；d_1 = 14.469；d_2 = 6.250；order_consistent = True
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: Photosynthesis takes the energy of sunlight and combines water and carbon dioxide to produce sugar and oxygen as this?
- r_o: waste product
- r_s: asexual reproduction

## SciQ e13bf18f9078…

- k = 10.438；d_1 = 14.969；d_2 = 5.906；order_consistent = True
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: What is another term for flagella?
- r_o: pseudopods
- r_s: air pressure

## PopQA deaf185e6af5…

- k = 23.641；d_1 = 25.000；d_2 = 22.281；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: Who was the director of The Sting?
- r_o: George Roy Hill
- r_s: Joe Berlinger

## PopQA 6aeb293f61d7…

- k = 23.109；d_1 = 23.656；d_2 = 22.562；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: Who was the director of The Game Plan?
- r_o: Andy Fickman
- r_s: Wolfgang Petersen

## PopQA c614af97e817…

- k = 22.906；d_1 = 23.750；d_2 = 22.062；order_consistent = True
- y_SS = 0（y_SS=0（正确接受）但 k 高（偏 r_o））
- Question: Who is the author of Evil Under the Sun?
- r_o: Agatha Christie
- r_s: Daniel Defoe

## PopQA a25350bab0b0…

- k = -3.500；d_1 = -2.500；d_2 = -4.500；order_consistent = True
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: What genre is Reality?
- r_o: rock music
- r_s: disaster film

## PopQA 55e538560f96…

- k = -2.344；d_1 = 6.812；d_2 = -11.500；order_consistent = False
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: What is Delhi the capital of?
- r_o: Lodhi dynasty
- r_s: Tuscany

## PopQA 1c96f0ba7877…

- k = -0.375；d_1 = 6.188；d_2 = -6.938；order_consistent = False
- y_SS = 1（y_SS=1（错误拒绝）但 k 低（偏 r_s））
- Question: What is Delhi the capital of?
- r_o: Khalji dynasty
- r_s: New South Wales

