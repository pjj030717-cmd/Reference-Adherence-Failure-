# token_span_mapping_audit.md

## 方法

1. `apply_chat_template(tokenize=False, add_generation_prompt=True)` 得到 rendered prompt；
2. 同一 tokenizer `return_offsets_mapping=True`；
3. offset-tokenization ids 与 `apply_chat_template(tokenize=True)` 完全一致（逐输入核对）；
4. R_end/C_end 由 rendered 中 Reference/Candidate 字段的字符 span 末尾定位；D_pos = prompt_len-1。

## 审计结果（随机 30 train + 30 dev）

### dev（30 groups）
- `4271cfe8b61a8a3393e85c06aa38d43ea9b075538d57c363e9ce3260328b75b4` ref_span=539:549 ('geotropism') cand_span=569:594 | R_end tok#98='ism' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `bb16e19b3e8abd378a0f469eb8c62aade0d44c020067925db7aa302329933e1f` ref_span=561:575 ('input distance') cand_span=595:624 | R_end tok#100=' distance' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `10c688835407c6795fb879cb2535d55f0927bb85029e051e3aa4de6c67c47398` ref_span=595:600 ('breed') cand_span=620:640 | R_end tok#107=' breed' C_end tok#116='.\n\n' D_pos tok#123='\n' | ids_ct==ids: True
- `3f7112327a3cfef5de23e93e4a1880826ae70b39c367298aad1f40d9c3d98af2` ref_span=634:646 ('interkinesis') cand_span=666:693 | R_end tok#114='inesis' C_end tok#125='.\n\n' D_pos tok#132='\n' | ids_ct==ids: True
- `41972e19b17353b471daa83ca263985b90734a684c6cf04de158ceba557bba20` ref_span=556:564 ('chordata') cand_span=584:607 | R_end tok#100='ata' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `615e53d88676086fc33a4ec1643b3b9372f1df9941c8919af9533b858c168cb5` ref_span=541:556 ('radial symmetry') cand_span=576:606 | R_end tok#99=' symmetry' C_end tok#109='.\n\n' D_pos tok#116='\n' | ids_ct==ids: True
- `d3f802c5c147cd1fb842715b1ad21f6f8e9faa2d9d00ce60d1fce94c4304ff1e` ref_span=561:567 ('darwin') cand_span=587:608 | R_end tok#101='win' C_end tok#111='.\n\n' D_pos tok#118='\n' | ids_ct==ids: True
- `9a95b89b7aed4069296c76ff091de2e4b363583c73a648a5c37edbc80ebe97ee` ref_span=635:655 ('atomic energy levels') cand_span=675:710 | R_end tok#115=' levels' C_end tok#126='.\n\n' D_pos tok#133='\n' | ids_ct==ids: True
- `53a9a275a436d8bd4aa6f496fd454816a1ff515b21770e062f15a0df2cb4322d` ref_span=572:577 ('rocks') cand_span=597:617 | R_end tok#113=' rocks' C_end tok#122='.\n\n' D_pos tok#129='\n' | ids_ct==ids: True
- `5da4110576e81a798a7c9eb3d4e2c0637adfdb17a24836f5997ee49a554c1edd` ref_span=587:608 ('alkaline earth metals') cand_span=628:664 | R_end tok#107=' metals' C_end tok#119='.\n\n' D_pos tok#126='\n' | ids_ct==ids: True
- `273e7a2bbe79fb45e3da5b01c63233ca2a0293e88be678371d45e6e734eb3687` ref_span=577:583 ('medium') cand_span=603:624 | R_end tok#103=' medium' C_end tok#112='.\n\n' D_pos tok#119='\n' | ids_ct==ids: True
- `6215cf61f682532748483804f0bec679f57e2deeb2019bd3c4e390f10a27f94f` ref_span=712:722 ('speciation') cand_span=742:767 | R_end tok#137='iation' C_end tok#147='.\n\n' D_pos tok#154='\n' | ids_ct==ids: True
- `3ee096de72cb3b0dfd08b81812498b3c90600256873b702dfe6c8228f2c9c0a4` ref_span=581:588 ('kinetic') cand_span=608:630 | R_end tok#105=' kinetic' C_end tok#114='.\n\n' D_pos tok#121='\n' | ids_ct==ids: True
- `6dd3c98e72585bcf228341dc2750f03d27c670e98803ecd3ede73796a9feeaf1` ref_span=525:531 ('alloys') cand_span=551:572 | R_end tok#94=' alloys' C_end tok#103='.\n\n' D_pos tok#110='\n' | ids_ct==ids: True
- `24b91442961df4239a123354c44e53dab4a44f7cff6e66fec314c22e1b1540ea` ref_span=552:558 ('cancer') cand_span=578:599 | R_end tok#99=' cancer' C_end tok#108='.\n\n' D_pos tok#115='\n' | ids_ct==ids: True
- `41f4f125f31ef8e2c24b2b54d17e9ec1332933a5182bb658df2b71e74210f07a` ref_span=530:537 ('equator') cand_span=557:579 | R_end tok#98='ator' C_end tok#108='.\n\n' D_pos tok#115='\n' | ids_ct==ids: True
- `a2aed8ae91384128577f7d97b413b47249fa8eea55b7860ee373095c2671a43d` ref_span=578:600 ('heterocyclic compounds') cand_span=620:657 | R_end tok#111=' compounds' C_end tok#123='.\n\n' D_pos tok#130='\n' | ids_ct==ids: True
- `801ea8780af5ab0b2c3cc77a3f4ac2b6c0bcddac3942f28db77284b3fab8c8b0` ref_span=578:587 ('the shell') cand_span=607:631 | R_end tok#103=' shell' C_end tok#113='.\n\n' D_pos tok#120='\n' | ids_ct==ids: True
- `199078912aef9601d43fbb41eeeb0ec45ab7f752eb6378b83eae3b082379161c` ref_span=556:569 ('cell membrane') cand_span=589:617 | R_end tok#99=' membrane' C_end tok#109='.\n\n' D_pos tok#116='\n' | ids_ct==ids: True
- `3dd056aadf1ba84092fcc9b21ef87dd26564c3486de97f4306b599e30773fdc6` ref_span=565:573 ('adhesion') cand_span=593:616 | R_end tok#103='hesion' C_end tok#113='.\n\n' D_pos tok#120='\n' | ids_ct==ids: True
- `9a48b8a8d35c10c9696fa88651a5e839600948a47294a46721caa270613cc5cd` ref_span=569:577 ('stronger') cand_span=597:620 | R_end tok#101=' stronger' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `788c34ef73e5ccaf4376fc748a0275e60532beed1cab5c8e3bcc7dd363c8537c` ref_span=584:601 ('population growth') cand_span=621:653 | R_end tok#110=' growth' C_end tok#120='.\n\n' D_pos tok#127='\n' | ids_ct==ids: True
- `23b2da192ff61c4753bc05910769816eed0a3667db2a4b773622bce58081ef84` ref_span=608:624 ('causes pollution') cand_span=644:675 | R_end tok#109=' pollution' C_end tok#119='.\n\n' D_pos tok#126='\n' | ids_ct==ids: True
- `8e63f950b177c6f1acee25e35005d2a081fbdc51abe07d934642ad1e7b23256b` ref_span=562:568 ('alloys') cand_span=588:609 | R_end tok#102=' alloys' C_end tok#111='.\n\n' D_pos tok#118='\n' | ids_ct==ids: True
- `50e89b0ddd95fe081a170155268d88b5637809604696c7a6f1e8c642f2b54c5a` ref_span=552:558 ('cancer') cand_span=578:599 | R_end tok#99=' cancer' C_end tok#108='.\n\n' D_pos tok#115='\n' | ids_ct==ids: True
- `d7bf00bba92da939b45008d97af6b4c84e6278724e013e42d7b4e27c38110458` ref_span=577:586 ('myoglobin') cand_span=606:630 | R_end tok#104='oglobin' C_end tok#114='.\n\n' D_pos tok#121='\n' | ids_ct==ids: True
- `3832009ffefb8053acc8f41b409ffc903ba02cb06ee1f396131afe3a64f7f714` ref_span=567:584 ('activation energy') cand_span=604:636 | R_end tok#102=' energy' C_end tok#112='.\n\n' D_pos tok#119='\n' | ids_ct==ids: True
- `52aba3c550dabe11b7e047ff5a4addc223e7b2d24ee14924ca69cfc2dae3b724` ref_span=572:579 ('impulse') cand_span=599:621 | R_end tok#102=' impulse' C_end tok#111='.\n\n' D_pos tok#118='\n' | ids_ct==ids: True
- `1a07b80e58d0120531bcefee7ba132e3bd77bef834bc592741b579d8a1952ad2` ref_span=530:559 ('replace another in a molecule') cand_span=579:623 | R_end tok#98=' molecule' C_end tok#111='.\n\n' D_pos tok#118='\n' | ids_ct==ids: True
- `d716b45450cc92f0207e397647f2e955120b268a9c3a06f632f6beb374da68a3` ref_span=611:620 ('electrons') cand_span=640:664 | R_end tok#114=' electrons' C_end tok#123='.\n\n' D_pos tok#130='\n' | ids_ct==ids: True

### train（30 groups）
- `f326a699f23afa8f831646fe0dc92f461e4e3a66cae0df65bae17e4c9eadf901` ref_span=559:572 ('vesicle coats') cand_span=592:620 | R_end tok#102=' coats' C_end tok#113='.\n\n' D_pos tok#120='\n' | ids_ct==ids: True
- `9e897ad9ea552e2431a2f4afeb9eead9ac799389d04f99d506410ef6d574eb4b` ref_span=577:596 ('hydrogen bonds form') cand_span=616:650 | R_end tok#106=' form' C_end tok#117='.\n\n' D_pos tok#124='\n' | ids_ct==ids: True
- `3d428c1a11fe7a27909a8ce088beab1fce82a257a9602616f84c2cd97998902a` ref_span=557:572 ('a virtual image') cand_span=592:622 | R_end tok#101=' image' C_end tok#112='.\n\n' D_pos tok#119='\n' | ids_ct==ids: True
- `3f49dd2593eeda0a0910e32363132d29fea6cb68fbe5f67c5f5a6596f5d8ce4a` ref_span=572:578 ('energy') cand_span=598:619 | R_end tok#101=' energy' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `6ef13bd6f84f712c13d2252aa3b57cb5a74433fd2638221b032ec7dafb1ab7aa` ref_span=576:591 ('indeterminately') cand_span=611:641 | R_end tok#106='ately' C_end tok#117='.\n\n' D_pos tok#124='\n' | ids_ct==ids: True
- `f83f1662f387b4ed4a3332094c62ee435d88de869b2037b1bee22c9741efeea6` ref_span=582:586 ('head') cand_span=606:625 | R_end tok#103=' head' C_end tok#112='.\n\n' D_pos tok#119='\n' | ids_ct==ids: True
- `cde50b66dbfee0a5ba313e58cbcc0f5223142a132d7f24597b1ad3988fabc512` ref_span=579:589 ('alteration') cand_span=609:634 | R_end tok#105=' alteration' C_end tok#114='.\n\n' D_pos tok#121='\n' | ids_ct==ids: True
- `0d06aa7ea0b16a7237c78bb114db45bc59e571046db5def14ec4629d8f9aba07` ref_span=557:568 ('prokaryotes') cand_span=588:614 | R_end tok#103='otes' C_end tok#115='.\n\n' D_pos tok#122='\n' | ids_ct==ids: True
- `c1522a97e9505a99322a4fb1e50018a12020647184b3c37da27d2a2a3ce5d656` ref_span=611:619 ('adhesion') cand_span=639:662 | R_end tok#113='hesion' C_end tok#123='.\n\n' D_pos tok#130='\n' | ids_ct==ids: True
- `834bb3b4376613dc44c2388fe763e9b90f0714bd319b35d90fa69551582fc626` ref_span=549:555 ('cyclic') cand_span=575:596 | R_end tok#96=' cyclic' C_end tok#105='.\n\n' D_pos tok#112='\n' | ids_ct==ids: True
- `8f2abbec0a761684c5e5cf842a5b98844c3f9375ccbfe6038bb22ccf00ff9604` ref_span=588:599 ('centrosomes') cand_span=619:645 | R_end tok#111='omes' C_end tok#122='.\n\n' D_pos tok#129='\n' | ids_ct==ids: True
- `3fbc04fadc2b3be95f01e5b4adf2d7a129da7bb88a1f6a68a0a8d754056703a3` ref_span=577:586 ('endurance') cand_span=606:630 | R_end tok#104=' endurance' C_end tok#113='.\n\n' D_pos tok#120='\n' | ids_ct==ids: True
- `f57b91ca2d8c2293773128d0870ac5bcc7a4ed55f3ea0fddff1a5127568053ab` ref_span=546:555 ('reactions') cand_span=575:599 | R_end tok#99=' reactions' C_end tok#108='.\n\n' D_pos tok#115='\n' | ids_ct==ids: True
- `200d76c0f185ea138fec623085ea5468f73b552654c8f78de9b225c28c3e7d95` ref_span=530:539 ('nutrients') cand_span=559:583 | R_end tok#96=' nutrients' C_end tok#105='.\n\n' D_pos tok#112='\n' | ids_ct==ids: True
- `718c00e1b0a61d0fa6c7bde6adb0807a56f3006d17a4b92952335c715bfa2169` ref_span=549:559 ('suspension') cand_span=579:604 | R_end tok#98=' suspension' C_end tok#107='.\n\n' D_pos tok#114='\n' | ids_ct==ids: True
- `226ec047bfbc01ed41c8246bdbd41c1542c69e9cb56c329fc1a332f06ecc0b40` ref_span=569:577 ('ph level') cand_span=597:620 | R_end tok#100=' level' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `9e8fc8c889094af99ef930d11b9f8093d273d8af69445414a5c59119ad06fbb2` ref_span=528:534 ('mosses') cand_span=554:575 | R_end tok#96='es' C_end tok#106='.\n\n' D_pos tok#113='\n' | ids_ct==ids: True
- `423f17fcc961033dd3d2c98807b3832ac43a886a5f6a1843b6fec4fde9b7fb67` ref_span=590:599 ('increases') cand_span=619:643 | R_end tok#103=' increases' C_end tok#112='.\n\n' D_pos tok#119='\n' | ids_ct==ids: True
- `33cb0a7a3c94c59c52c6d8b6a28f3cc798129ca30eeabbc30dac4583241f1579` ref_span=547:553 ('volume') cand_span=573:594 | R_end tok#97=' volume' C_end tok#106='.\n\n' D_pos tok#113='\n' | ids_ct==ids: True
- `9bc2b27531180379163a65da8c60bf25874bc8def09176b6e25a511cd2dc0a5d` ref_span=603:614 ('sedimentary') cand_span=634:660 | R_end tok#109='ary' C_end tok#119='.\n\n' D_pos tok#126='\n' | ids_ct==ids: True
- `2ffd17bd9816b240e3e1ffaaec953b60a9e68c744744ed61fbf4d15370272195` ref_span=658:664 ('pollen') cand_span=684:705 | R_end tok#115=' pollen' C_end tok#124='.\n\n' D_pos tok#131='\n' | ids_ct==ids: True
- `665107ef31a82ea66ff15a81a18c34594ad31a64313cfd76d9d4764f47f847c2` ref_span=568:577 ('magnitude') cand_span=597:621 | R_end tok#101=' magnitude' C_end tok#110='.\n\n' D_pos tok#117='\n' | ids_ct==ids: True
- `f720c93adba8e32b9ece65f413b7a3b4d482aaf214d5dab0644e46386f1df922` ref_span=562:579 ('population growth') cand_span=599:631 | R_end tok#101=' growth' C_end tok#111='.\n\n' D_pos tok#118='\n' | ids_ct==ids: True
- `b619c7f920e0ffb8b74b41922fe2f515e2f1947f1468f5bfcf257d6ee427cd62` ref_span=635:649 ('shield volcano') cand_span=669:698 | R_end tok#113=' volcano' C_end tok#123='.\n\n' D_pos tok#130='\n' | ids_ct==ids: True
- `810b5890647ba3ed3af06236881b16aecc790a9d2e6ce0aed2f8a3583cab2653` ref_span=539:552 ('metamorphosis') cand_span=572:600 | R_end tok#97='osis' C_end tok#108='.\n\n' D_pos tok#115='\n' | ids_ct==ids: True
- `c40827198415db2cc17438b1ac031e39eacc4d25820a68d0bc2d4dbd43a18c5d` ref_span=629:649 ('asexual reproduction') cand_span=669:704 | R_end tok#118=' reproduction' C_end tok#129='.\n\n' D_pos tok#136='\n' | ids_ct==ids: True
- `b554aca42018277c7ba19b9a651e2e6878484125b8cdf51b27a77ed397438383` ref_span=593:602 ('asteroids') cand_span=622:646 | R_end tok#106=' asteroids' C_end tok#115='.\n\n' D_pos tok#122='\n' | ids_ct==ids: True
- `7994c220ee83b28dc206f11428b0002d2f150cb759cae7cd1aa4c7137f9fdee8` ref_span=583:599 ('chronic exposure') cand_span=619:650 | R_end tok#104=' exposure' C_end tok#114='.\n\n' D_pos tok#121='\n' | ids_ct==ids: True
- `61bbba116f417867fbde83d3a164d585a38eb791ecfaf65f163009b1eb550d21` ref_span=562:577 ('plasma membrane') cand_span=597:627 | R_end tok#99=' membrane' C_end tok#109='.\n\n' D_pos tok#116='\n' | ids_ct==ids: True
- `d74c4879c732b1227d9a8a43c0f218a3a82b06f4ef7da4d0bef58910d3681e71` ref_span=598:610 ('fossil fuels') cand_span=630:657 | R_end tok#106=' fuels' C_end tok#116='.\n\n' D_pos tok#123='\n' | ids_ct==ids: True

## 结论
- offset-tokenization ids 与 apply_chat_template(tokenize=True) 逐输入一致：通过
- R_end 均落在 Reference Answer 正文最后一个非空白 token（非字段名/换行/Candidate/Answer:）
- C_end 均落在 Candidate Answer 正文最后一个非空白 token
- D_pos = prompt_len - 1 恒成立
- token ids 与 span 对齐唯一确定

状态：`token_span_mapping 有效`
