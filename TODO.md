- ✅ Ensure app.py audio input sounds like training data
- ✅ Use a huggingface transformer with the dataset
- Verify that the training spectrogram matches the predict spectrogram
- Count number of example misses in dataset loading
- Verify windowing and jitter params in Song Dataset
- Create an attention-based network
- ✅ Increase parameter count in network
- Verify that labels really match what is on the music4dance site
- ✅ Read the Medium series about audio DL
- double check \_rectify_duration
- ✅ Filter out songs that have only one vote
- ✅ Download songs from [Best Ballroom](https://www.youtube.com/channel/UC0bYSnzAFMwPiEjmVsrvmRg)

- ✅ fix nan values
- Try higher mels (224) and more ffts (2048)
- Verify random sample of dataset outputs by hand.

- Train with non music data and add a non music category
- Add back class weights
- Add back multi label classification
