Imagine such a scenario
=======================
  
One day, you'd like to build a story online through materails: __Image__ (main component), Text, etc. At the begining, you can input some _descriptiions_ or pick up some _image_s you wants. Then, based on this information, the engine will show a running __gallery___ on the top of the website. It can _memorize_ what you pick or even what you pick at the beginning, and throw away later on. It seems that the engine can __Read Your Mind__ by just analyzing the images you pick and show what you want for the next. It's about __image understanding__, similar to [text understanding](https://github.com/Chasego/ubw/blob/master/labs/papers/nips/2015/Teaching%20Machines%20to%20Read%20and%20Comprehend/1506.03340v1.pdf). It's also similar to the Amazon recommender engine, but totally not the same.
  
Back to the implementation question, "__What are techniques maybe useful for this project ?__"

+ _LSTM_, for memorizing what images you picked

+ _CNN_, for figuring out the patterns. However, it's far from enough. Since what I want to achieve is analyzing the "__ELEMENTS__" in the images, and the way they associate with each other. The "__ELEMENTS__" are not necessarily the "ROI"s, however, should be the key points, like the color patterns.

+ More
