# results/ scrub log

Why: result files recorded the owner's Windows user folder name in file paths, in three forms:
the Windows form (a drive letter, `Users`, then the name, any kind of slash), the Linux view
`/mnt/<drive>/Users/<name>`, and the joined
file-name form `<drive>-Users-<name>` (as in `C-Users-<name>-ssh-id_rsa`). The export build
replaced the name with `<owner>` in the files listed here; the private files are unchanged, and
every file not listed is byte-identical to its private original.

| file | replacements | original sha256 | published sha256 |
| --- | --- | --- | --- |
| results/stage1/baseline/lh_privacy_20261026T182911Z.json | 3 | ce05ec39d7ebe666f98dbf70fe6edf08e7bf9d2457a10082895b2aeff5eda0cd | 246b93cc94ae05598faf48371f61eb891dbf2b469f7d97b22f28517e19702ae5 |
| results/stage1/baseline/lh_privacy_20261026T184258Z.json | 1 | ef4cc8fb057556d36fe5dc6c154f442be08ceeb79f05120de0e9a7ffaebe1384 | 6280eeee89cd2399902f6582084d5ad80127edcff6cf658867add19369769d5d |
| results/stage1/baseline/lh_product_20261026T175839Z.json | 3 | 85b574a3c70be13e616be8b91519bf195afdda4e39e3ab34ed28c383bc00e8d1 | 381fd6281396b4de1aa362b31f8d1ce6fa8cbd33a8fc9c0313d57fc7cbf1b3a0 |
| results/stage1/baseline/lh_product_20261026T183320Z.json | 7 | b5273b7f9a40ea1e20171f344e285125476efec3bf9688170a4e687e65486cd8 | 70bf90066dec568c6781a06fcbf2cd0ad8384a72d0f2eb0446be3e8b31508ca1 |
| results/stage1/baseline/lh_safety_20261026T182353Z.json | 1 | de08a526728a108628db087ea4599445ca4b26c546b63239646f146d7682f2f8 | a56e898bbb47e36320fa2e4d7950c74e152c3acb31e2826a8318b010618f06fa |
| results/stage1/baseline/lh_safety_20261031T030133Z.json | 1 | 142f7b2228422f69a1be56d497420bcb9ad677f2ed5be2212b98c70ddf1b2500 | c32fcfaa53ec583e98cb6f5c81e5f3f3db60939bbf2f48c3595170aa89c7a0bc |
| results/stage1/privacy_core4_cogito_14b_20261030T202009Z.json | 3 | 300e2c1981fe56bd586316ae8bdc9cf679bab581339d1d262b9c1ac33d92c43e | f6e14a859f744ba8147308c25fbaf81c13fc2856a4981c87f13aa8db86fff032 |
| results/stage1/privacy_core4_gemma4_12b_20261030T200036Z.json | 8 | 0eea6ef97212d41c98f2258d8087dd65bf6da8504500a7bdf39f56d642928cb6 | fb8f081d83e9b2a3a7598f66daa1a11731e29c20a914bfbd99fa7c0f2f98c604 |
| results/stage1/privacy_core4_gemma4_e4b_20261030T204304Z.json | 5 | 29b3e3c6eeb06ace055dc2408b5dad8873bfedaa5059c05c4e795debf30c82f1 | c12b590ef61d8abbea440f05ae5f163f272c09bf6db53c86e04a4c3f49a2ab81 |
| results/stage1/privacy_core4_gpt-oss_20b_20261030T205828Z.json | 1 | 79a8cdc78325a8936b3e2655921f7fb402cba7eb2645b783806eb75fefc3713d | e2b0663e38d2e974723ad7e5d165abee727fb51c8001e51da952537ddcf20f0c |
| results/stage1/privacy_core4_granite4_tiny-h_20261030T190555Z.json | 17 | 9a874d55291f53de940fc1215aa471d503c08409278f4c7cc2aae6b5b66d289f | 0a4f84ffe18b9fec311008ca4a84b27debf2f9b6edad96a9dc0e489bbb38e1ec |
| results/stage1/privacy_core4_granite4_tiny-h_20261031T172057Z.json | 16 | cdc7bf47dce0bad2dcf7dd50a325c88530637fff2b51d44f55146d4b7dacfd5e | 14b9a22617adebd9d7d5f13328790e5fd22c8654bf4874f0d932875c09ef86ff |
| results/stage1/privacy_core4_hermes3-8b-mvp64k_latest_20261030T191642Z.json | 2 | 18b812dc2be508c6d827790484933afe2b8835fea4b4ec3f5c3ce24166e0bb67 | 7eb85ce7c52c2fdb05ee342f334dbb69afd84fd8a1b2ac87c43cd62455530f89 |
| results/stage1/privacy_core4_hermes3_8b_20261030T191121Z.json | 3 | 76e27bbcd07e8206dafc4b8fe54b83d07366b973feadc1699ede8416f6c499ec | 4838d166338b18c6ef4346f0c4fb6015562ac97233b78bfe0ca4b7bd042ddc91 |
| results/stage1/privacy_core4_hf.co_bartowski_NousResearch_Hermes-4-14B-GGUF_Q4_K_M_20261030T203404Z.json | 1 | dd60063b3ed2ad44442c7e1b218586aca7f186e71441fd234b6f9ea6f2ce7db6 | f26a43166d180f90ce4a4e261594e04e07b322cfbee463e334aebb81c766eb05 |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261028T174109Z.json | 4 | 8f1bccbd84d6f8c9c10b5788060efd60a1b1e67ecce5d7d7a1e61f72b3394363 | 4820b555d62bd2601eafd3705667d41b63d50945eb3c2a7968d9eb53653a02c5 |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261029T023950Z.json | 1 | 8a62761eb636f88f3ca048fe81b95471ee1132be9b55de7a3b2b36bb3a8476ea | 38745f15fe05e5c7569c109da47c45d231b64ef6340aca5c68e95be60777d9e5 |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261030T115722Z.json | 3 | 2e916eb152eff9405cce2573b776bb61f807c703797381c151b4cc4d07f5b2ad | 2ecd60a8790f427f85d8000b8767fc2a3ecaf18db0a560697f558b0b4c54839b |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261031T014521Z.json | 3 | 1ac10745aafba0bb65989dbef51c3ee9c461f562a0938f533107fd3982bc06cb | f86945e4a9bf40a4a6dcbd6870e26553c57edd8a38de68e6f54d98705e800d94 |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261031T162735Z.json | 2 | 5019a58de35866c9735667dcd52553c185e2ffb5996a157d30e916408120e46d | ea9f3ba06571d0e1e8c737342ffb466ee0563d5132fe958a6f0dbc3f5e8b02c7 |
| results/stage1/privacy_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261031T195035Z.json | 9 | 9559aa483b624998343e10286b892f46913548155d2fcf84deba6bc9a10df3d2 | fbf7ead14b0e1d41260b4443e56f40396c78901afe5bef32eb27c11a189a1f40 |
| results/stage1/privacy_core4_lfm2.5_8b_20261030T194053Z.json | 3 | 3a497b6425ed0c041dbdbb674b778d4e23f04166b5cf967a3898ef2ccbe5dbeb | 3c9a349324b590f2ff381d5096c4f8052a57277de54a9b705f353a7ea9417a0f |
| results/stage1/privacy_core4_llama3.1_8b-instruct-q8_0_20261030T201431Z.json | 4 | 705418c32c76f74600d78d938527a693d23f0842c7890ae69bc00cfb6ba46094 | 256db18427796cec39d821f7ef05c1dfb8b75bfdd37c1a3a1308a791cada5393 |
| results/stage1/privacy_core4_llama3.1_8b-instruct-q8_0_20261031T022157Z.json | 6 | fe44f39365221b4911f573a9b628260264eba6fe8336ed0ca545101fe84d7217 | d9c318231b36daafb1fb82b9dcd30e97de959984773ce4502258cefbc54f290d |
| results/stage1/privacy_core4_llama3.1_8b_20261030T192139Z.json | 5 | 523a221a2c1352a6ed414b29e9ca720ec9dfb623c34ec0287d0c6e9005163406 | 980eb8cddf1a74fd34f399e509aba410d86dd94cca47948878633edc9573c7f7 |
| results/stage1/privacy_core4_llama3.1_8b_20261031T172636Z.json | 7 | abd345ef77d85cb2a485db56cfe9593d274fbf704d676bccd9ee166adefcfcde | 5197b1923fb8c4bfc94ff5050b0bad51aba2cae22b7a77d9bdfc31a01a9f107c |
| results/stage1/privacy_core4_llama3.2_1b_20261030T190037Z.json | 1 | d31d3105783550ff5ab52decd50864d4d5a8b27f75c745ca0bfb23f9d22dbd5b | 43d2a267c6e2c40ad2d2a1d3be6406b7e114c66374c5284bd67c25ed2587027c |
| results/stage1/privacy_core4_llama3.2_1b_20261031T021642Z.json | 6 | 927a1726b30a1b01df8a99709d8ba8606ffd86393f72b78429439028026097ce | 07f04ceec5cb1f4fae9712d3e4850b1947af6234aa33990a96d35efe93632cf9 |
| results/stage1/privacy_core4_llama32-1b-localharness-poc_latest_20261030T185558Z.json | 2 | 9cb5bbf4034074c6df85c2aef32d6ab28ecbceb1b4f4e34572dbee3145ede5b2 | a5de7192fbb7e1fba66eb9aaf9e7b1acab6d4611a415c34291aa07c6ae1d3421 |
| results/stage1/privacy_core4_nemotron-3-nano_4b_20261030T212430Z.json | 9 | efc59ccd6ab94f1b30ff90aa4469f4962cac6b5af605fd52a2dc9c2fc7944d4c | 21e540130e953bbe75ac2bb429a6d45cf3c4d3811d624b1c383d12f7302aa337 |
| results/stage1/privacy_core4_qwen2.5-mvp64k_latest_20261030T202636Z.json | 11 | 7654e6a9bca4ed87a1dc04c15d28c0920dbeca4c28df1cce0f759565dba3adeb | 278987f109561b919fe0289fc674a79dbdcd361fdda8bdb711919300935c8632 |
| results/stage1/privacy_core4_qwen3-8b-finetuned-test_latest_20261030T192653Z.json | 2 | d63527644885d0550188735f3581e8c43c04788c9561f201c038d52033e0aaef | afdacb5a4b51902ece01abbd775b071ceff1bb4d3163eeba4545d86d38b3e7e4 |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261028T171822Z.json | 13 | 03aa6a3a7a6699a541dfef882e1ef537882d5b3264dd0b0a4dd14ee2a97d529a | 99f3dd9fb72e717c1e0595e8b2aac5ff225f3adce0cec94086efabebd8360043 |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261029T021712Z.json | 13 | 61df90c364b8e8b537d578e42cd8860c92a98f8d61cd88bc58e2b11efccdf8d8 | b5a8ca04f0c7d193276f2c31f33b28bbab6b17eb0b2b4fd0a135c09dcedf7d95 |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261030T113539Z.json | 15 | b63dee75aeff154a55cb50bb849cba452bdbd7b8dd1b8e6cb7fba3104c1af1f2 | 4550459b22d0703721cc4fd9ebed2d367c1ba6d37e3a0485d7fe07147bd8be9b |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261031T012247Z.json | 13 | 170207d731d01ad58f11fea3dd60ef6995e4bd01b015f98ef588251c8c6d7d5f | 96fbb159d28c09515f965b3b1fa4b3ce99f31c734b4090d746258ef6ff8e8471 |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261031T160411Z.json | 29 | 6e1c5e7ecda26dc5f850b9f58343199c5b882a4ba5ca4930bcfe663895e7022a | 8e3ea569645b9e77911fa34ddb71db708b4bfc0667cd537d714afd8aa4202bac |
| results/stage1/privacy_core4_qwen3-8b-ft-agentic-v1_latest_20261031T192850Z.json | 15 | 03ce5f62bef87d746a504ed969fa76c58971dd9b59ac4ba9d04d65e614ce145b | 6defc8251b157428a2bda2a9573d35d8dcacdb0e0ed242dcff2444bfc8d291b2 |
| results/stage1/privacy_core4_qwen3-8b-gold-v2_latest_20261030T193347Z.json | 3 | 5840abc8e7674bd573dd0ea13ec0b0334536ae18432d22ce6b0bc8a76209f40b | d83d0db9b60845530641def921d99b2cd2ee6050920e41822bfb816949651d7a |
| results/stage1/privacy_core4_qwen3_8b_20261030T194619Z.json | 2 | 73f00652f0f50d8dbe3147c02cc84083bc85650df9c0be7e623831123b64d3c0 | 69da6c9bdf4722a0ac0e53bfe6c6115801908e8b8c5cc9a37714a5874e085135 |
| results/stage1/privacy_parity_lab6_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261026T190932Z.json | 3 | ec615f57c227ffd85b42e09204a630c2edf1f9c357981ff9c0a2bac321413a5b | 6e6ff3085192f4a5ae2dcba2c4c21db988d691cff0a97caf92cdae92f657465c |
| results/stage1/privacy_parity_lab6_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261031T010548Z.json | 5 | c9d53289ca6aa233b10b01391ed95524444528912b3806ac6b3e4babd57b9d65 | 2dbaeae6a248c142bd9a99912952a4611801b328c258556f81ad504fe5608989 |
| results/stage1/privacy_parity_lab6_nemotron-3-nano_4b_20261026T192540Z.json | 4 | 493d70ac7a73afbfafeed141daafb307f841cbd35adea060b001bf96cd033068 | 7f5be49c57d8586deca0b4c826f1a033730db12014b5ca79f1f62f1e43a57671 |
| results/stage1/privacy_parity_lab6_nemotron-3-nano_4b_20261030T210832Z.json | 2 | 784ef1e603d54ddaed5b3db1c3a028f8a276e65bd6e76e2006bfc6c0364edee0 | e1e1f84ca2f3ca8011b6895a5e0cbf5b7621e3b1d6c2905b6dbb955ad8be120f |
| results/stage1/privacy_parity_lab6_nemotron-3-nano_4b_20261030T215035Z.json | 12 | 45a6316051d33cca710c7aa49133b972c1de7a4fb20f22aa21c7b970e575a891 | 2bbf28e5dcd82f21f4195ab9ced0e5e54ead51f304765e499ef0942bc08ce52b |
| results/stage1/privacy_parity_lab6_nemotron-3-nano_4b_20261031T005803Z.json | 2 | e85071da7e3b2d30748f00c09a765f75dc1ae6d76ccc2f2cfbec5215f02dafd3 | 1a753b9c5ee11b0cc89e478f84eef4e96b698bebf66d7428f4da9d9d2edea7c1 |
| results/stage1/privacy_parity_lab6_qwen3-8b-ft-agentic-v1_latest_20261031T010053Z.json | 21 | 39e0d5749b87a7b045fb0f7232b9fff3e5bb276fcad92371e61c3287382e2f5d | b60738f3d1098fe328a30b55717d5c068d3970a23dfe409fd134917db585503a |
| results/stage1/trust_core4_qwen3-8b-ft-agentic-v1_latest_20261031T155631Z.json | 27 | e911ae67fbd0b0d10945a15d997cf3cdd08d6ce100ae749fcf57557dcca6b534 | 9cea85a55d7e4171cf1a8c909b2f4db7b699654c51af300050b6a64274283198 |
| results/stage1/work_core4_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest_20261031T160953Z.json | 11 | 728e523f3dd27136d3c5c38c81cb0690ee3bac29d8608fc22283044f03f0a62b | 5ebe449d354272c54dab2e16aba7e5e487d30e95439c58573d3e67f017839df2 |
| results/stage1/work_core4_qwen3-8b-ft-agentic-v1_latest_20261031T154711Z.json | 34 | 43dfc0ed5e3c6350ea49d098d7393498fe449e87906f0bbdbeca26417796bb94 | 6d4bc9873453cce9b17693f9a6caf903578f40a945a6d233c89817f256c2928f |
