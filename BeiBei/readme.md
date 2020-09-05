## Group-Buying Recommendation Dataset From BeiBei

---

### Dataset Files
- `data_size.txt`
  - Introduction: the count of users and items.
  - Format: \<#user\>\t\<#item\>
  - MD5: 9bdec95708f09bed98a077675df794ab
- `valid/invalid.txt`
  - Introduction: all successful (valid)/failed (invalid) UNDIVIDED records. *leave-one-out* is applied to `valid.txt` to generate `train_without_invalid/tune/test.txt`.
  - Format: (\<user_id1\>\t\<item_id\>\t[\<user_id2\>, ...]) for each line, it's IMPORTANT to note that the first user_id is the initiator's ID and the remaining user_id are the participants' ID (if exists).
  - MD5: 
    - 0c22be048a54a7592baf8e1054d4c298	valid.txt
    - 26bb46088f9d6a4dc18aabb94cb832e5	invalid.txt
- `train_without_invalid/tune/test.txt`
  - Introduction: the positive records for training/tuning (cross-validation)/testing phase.
  - Format: the same as `valid/invalid.txt`.
  - MD5:
    - 4a881989e3644925fe67629d0a897cec	train_without_invalid.txt
    - 5a4807f6329a8b0860e8a47ecaebf4b5	tune.txt
    - 9187a80d67f68830e3daec130979aeaa	test.txt
- `train.txt`
  - Introduction: both successful and failed records for training phase, that is, the combination of ``train_without_invalid.txt` and `invalid.txt`.
  - Format: the same as `valid/invalid.txt`.
  - MD5: 94589c25681d07bf013de7960ef2d8f1
- `social_relation.txt`
  - Introduction: the social relations among all users.
  - Format: (\<user_id1\>\t\<user_id2\>) for each line, meaning that the two users are linked in social networks.
  - MD5: b7b8ad477408193570c80de8073b9a9d

Due to the file size limitation, the negative sample files corresponding to `tune/test.txt` will be uploaded to GitHub Releases. Each line of the negative sample files is 999 user_id separated by \t and corresponds to `tune/test.txt` by line number.  

### Citation

If you want to use our dataset in your research, please cite:

```

```



### Acknowledgement

