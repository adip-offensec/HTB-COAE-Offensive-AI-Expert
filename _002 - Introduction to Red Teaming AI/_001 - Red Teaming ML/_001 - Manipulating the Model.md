Now that we have explored common security vulnerabilities arising from the improper implementation of ML-based systems, let us take a look at a practical example. We will explore how an ML model reacts to changes in input data and training data to better understand how vulnerabilities related to data manipulation arise. These include input manipulation attacks (ML01) and data poisoning attacks (ML02).

We will use the spam classifier code from the [Applications of AI in InfoSec](https://academy.hackthebox.com/module/details/292) module as a baseline. Therefore, it is recommended that you complete that module first. We will use a slightly adjusted version of that code, which you can download from the resources in this section. Feel free to follow along and adjust the code as you go through the section to see the resulting model behaviour for yourself.

---

## Manipulating the Input

The code contains training and test data sets in CSV files. In the file `main.py`, we can see that a classifier is trained on the provided training set and evaluated on the provided test set:

Code: python

```python
model = train("./train.csv")
acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")
```

Running the file, the classifier provides a solid accuracy of `97.2%`:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Model accuracy: 97.2%
```

To understand how the model reacts to certain words in the input, let us take a closer look at an inference run on a single input data item. We can utilize the function `classify_messages` to run inference on a given input message. The function also supports a keyword argument `return_probabilities`, which we can set to `True` if we want the function to return the classifier's output probabilities instead of the predicted class. We will examine the output probabilities since we are interested in the model's reaction to the input. The function `classify_messages` returns a list of probabilities for all classes. We are using a spam classifier that only classifies into two classes: ham (class `0`) and spam (class `1`). The class predicted by the classifier is the one with the higher output probability.

Let us adjust the code to print the output probabilities for both classes for a given input message:

Code: python

```python
model = train("./train.csv")

message = "Hello World! How are you doing?"

predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]

print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

When we run this code, we can take a look at the module's output probabilities, which is effectively a measurement of how confident the model is about the given input message:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Ham
Probabilities:
	 Ham: 98.93%
	Spam: 1.07%
```

As we can see, the model is very confident about our input message. This intuitively makes sense, as our input message does not appear to be spam. Let us change the input to something we would identify as spam, like: `Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF`. After rerunning the code, we can see that the model is now very confident that our input message is spam, just as expected:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Spam
Probabilities:
	 Ham: 0.0%
	Spam: 100.0%
```

In an input manipulation attack, our aim as attackers is to provide input to the model that results in misclassification. In our case, let us try to trick the model into classifying a spam message as ham. We will explore two different techniques in the following.

#### Rephrasing

Often, we are only interested in getting our victim to click the provided link. To avoid getting flagged by spam classifiers, we should thus carefully consider the words we choose to convince the victim to click the link. In our case, the model is trained on spam messages, which often utilize prizes to trick the victim into clicking a link. Therefore, the classifier easily detects the above message as spam.

First, we should determine how the model reacts to certain parts of our input message. For instance, if we remove everything from our input message except for the word `Congratulations!`, we can see how this particular word influences the model. Interestingly, this single word is already classified as spam:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Spam
Probabilities:
	 Ham: 35.03%
	Spam: 64.97%
```

We should continue this process with different parts of our input message to get a feel for the model's reaction to certain words or combinations of words. From there, we know which words to avoid to get our input past the classifier:

|Input Message|Spam Probability|Ham Probability|
|---|---|---|
|`Congratulations!`|64.97%|35.03%|
|`Congratulations! You won a prize.`|99.73%|0.27%|
|`Click here to claim: https://bit.ly/3YCN7PF`|99.34%|0.66%|
|`https://bit.ly/3YCN7PF`|87.29%|12.71%|

With this knowledge, we can try different words and phrases with a low probability of being flagged as spam. In our particular case, we are successful with a different scenario for the reasons outlined earlier. If we change the input message to `Your account has been blocked. You can unlock your account in the next 24h: https://bit.ly/3YCN7PF`, the input will (barely) be classified as ham:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py 

Predicted class: Ham
Probabilities:
	 Ham: 57.39%
	Spam: 42.61%
```

#### Overpowering

Another technique is overpowering the spam message with benign words to push the classifier toward a particular class. We can achieve this by simply appending words to the original spam message until the ham content overpowers the message's spam content. When the classifier processes many ham indicators, it finds it overwhelmingly more probable that the message is ham, even though the original spam content is still present. Remember that Naive Bayes makes the assumption that each word contributes independently to the final probability. For instance, after appending the first sentence of an English translation of Lorem Ipsum, we end up with the following message:

Code: txt

```txt
Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF. But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.
```

After running the classifier, we can see that it is convinced that the message is benign, even though our original spam message is still present:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Ham
Probabilities:
	 Ham: 100.0%
	Spam: 0.0%
```

This technique is particularly effective in cases where we can conceal the appended message from the victim. Think of websites or e-mails that support HTML, where we can hide words from the user in HTML comments, while the spam classifier may not be HTML context-aware and thus still base the spam verdict on words contained in HTML comments.

---

## Manipulating the Training Data

After exploring how manipulating the input data affects the model's output, let us proceed to the training data. To achieve this, let us create a separate training data set to experiment on. We will shorten the training data set significantly so our manipulations will have a more significant effect on the model. Let us extract the first 100 data items from the training data set and save them to a separate CSV file:

  Manipulating the Model

```shell-session
@htb[/htb]$ head -n 101 train.csv  > poison.csv
```

Afterward, we can change the training data set in `main.py` to `poison.csv` and run the Python script:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Model accuracy: 94.4%
```

As we can see, the model's accuracy drops slightly to `94.4%`, which is impressive for the tiny size of the training data set. The drop in accuracy can be attributed to the significant reduction in training data, which makes the classifier less representative and more sensitive to changes. However, this sensitivity to changes is precisely what we aim to demonstrate by injecting fake spam entries into the data set (`poisoning`). To observe the effect of manipulations on the training data set, let us adjust the code as we did before to print the output probabilities for a single input message:

Code: python

```python
model = train("./poison.csv")

message = "Hello World! How are you doing?"

predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]

print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

If we run the script, the classifier classifies the input message as ham with a confidence of `98.7%`. Now, let us manipulate the training data so that the input message will be classified as spam instead.

To achieve this, we inject additional data items into the training data set that facilitate our goal. For instance, we could add fake `spam` labeled data items with the two phrases of our input message to the CSV file:

Code: csv

```csv
spam,Hello World
spam,How are you doing?
```

After rerunning the script, the model now produces the following result:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Spam
Probabilities:
	 Ham: 20.34%
	Spam: 79.66%
```

As we can see, this minor adjustment to the training data set was already sufficient to alter the classifier's prediction. We can further increase the confidence by appending two additional fake data items to the training data set. This time, we will use a combination of both phrases:

Code: csv

```csv
spam,Hello World! How are you
spam,World! How are you doing?
```

Keep in mind that duplicates are removed from the data set before training. Therefore, adding the same data item multiple times will have no effect. After appending these two data items to the training data set, the confidence is at `99.6%`:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Predicted class: Spam
Probabilities:
	 Ham: 0.4%
	Spam: 99.6%
```

As a final experiment, let us add the evaluation code back in to see how our training data set manipulation affected the overall model accuracy:

Code: python

```python
model = train("./poison.csv")

acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")

message = "Hello World! How are you doing?"

predicted_class = classify_messages(model, message)[0]
predicted_class_str = "Ham" if predicted_class == 0 else "Spam"
probabilities = classify_messages(model, message, return_probabilities=True)[0]

print(f"Predicted class: {predicted_class_str}")
print("Probabilities:")
print(f"\t Ham: {round(probabilities[0]*100, 2)}%")
print(f"\tSpam: {round(probabilities[1]*100, 2)}%")
```

Running the script a final time reveals that the accuracy took only a minor hit of `0.4%`:

  Manipulating the Model

```shell-session
@htb[/htb]$ python3 main.py

Model accuracy: 94.0%
Predicted class: Spam
Probabilities:
	 Ham: 0.4%
	Spam: 99.6%
```

We forced the classifier to misclassify a particular input message by manipulating the training data set. We achieved this without a substantial adverse effect on model accuracy, which is why data poisoning attacks are both powerful and hard to detect. Remember that we deliberately shrunk the training data set significantly so that our manipulated data items had a greater impact on the model. In larger training data sets, many more manipulated data items are required to influence the model in the desired way.

1. Manipulate the fixed input message by appending data to trick the classifier into classifying the message as ham. Submit the flag you obtain after providing an input that satisfies the lab requirements.
	1. Go to http://94.237.61.52:57626/input_manipulation and then enter 
		1. Congratulations! You won a prize. Click here to claim: https://bit.ly/3YCN7PF. But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness.
	2. We get the flag `HTB{9b8de0fd17f2166743cd59f7ec876ac7}`

2. Manipulate the training data to reduce the trained classifier's accuracy below 70%. Submit the flag you obtain after providing a dataset that satisfies the lab requirements.
	1. Download the training set provided.
	2. Now use `head -n 10 training_data.csv > poison.csv` and change the python code to point to the `training_data.csv` file.
	3. Now in `poison.csv` the following text to it.

```
spam,Hello World
spam,How are you doing?
spam,Hello World! How are you
spam,World! How are you doing?
spam, Hello World! How are you doing today?
ham, I was just checking in to see how everything is going on your end since we have not had a chance to talk properly over the last few days.
spam, Hey there! Hope you are having a great day.
ham, After reviewing the notes from our last conversation I realized there were a few points worth expanding on so that nothing gets misunderstood later.
spam, Hi! Are you free to talk for a bit?
ham, I spent most of the morning organizing files and cleaning up old folders which turned out to be more time consuming than I originally expected.
spam, Hello! Just wanted to say hi.
ham, It might be a good idea to reschedule the meeting to tomorrow because several people mentioned they would not be able to join today.
spam, Hey! How have things been?
ham, While working on the task I noticed a few inconsistencies in the data that should probably be addressed before moving forward.
spam, Hi there! How is everything?
ham, I really appreciated your detailed response earlier because it helped clarify a few things that were not entirely clear to me before.
spam, Hello World! How are you?
ham, This message is a bit long but I wanted to explain the reasoning behind the decision so there is a clear record of why it was made.
spam, Hey! What are you up to?
ham, The traffic was unusually heavy this morning which caused a slight delay but everything else went according to plan.
spam, Hi! How is your day going?
ham, Once the update finishes installing we should run a few tests just to make sure everything is functioning as expected.
spam, Hello! Long time no see.
ham, I will send over the revised document later today after I double check the formatting and correct a few minor issues.
spam, Hey there! How are you doing?
ham, Sometimes writing a longer message helps avoid back and forth questions so I am including all the details here.
spam, Hi! Just checking in.
ham, Based on the discussion we had earlier I think it would be best to proceed cautiously rather than rushing into changes.
spam, Hello! Hope all is well.
ham, I reviewed the timeline again and it seems manageable as long as there are no unexpected delays.
spam, Hey! How is everything going?
ham, After thinking about it overnight I believe your suggestion makes sense and should be implemented.
spam, Hi there! What’s new?
ham, I was going through some older messages and realized how much progress we have made since we first started.
spam, Hello! How are you feeling today?
ham, Even though the task was challenging it ended up being a good learning experience overall.
spam, Hey! Got a minute?
ham, Let us reconnect later this week when everyone has more availability to discuss this properly.
spam, Hi! Hope your day is going well.
ham, The explanation may seem long but it provides important context that could be useful later.
spam, Hello! Just wanted to follow up.
ham, I noticed the issue you mentioned earlier and I think I have an idea about what might be causing it.
spam, Hey there! How’s it going?
ham, Once we receive confirmation from the remaining team members we can finalize the plan.
spam, Hi! How have you been?
ham, I wanted to write everything down clearly so there is no confusion about the next steps.
spam, Hello World! How are you doing?
ham, Taking a bit of extra time now should help us avoid unnecessary problems in the future.
spam, Hey! Just saying hello.
ham, I appreciate your patience while we work through these remaining details.
```

The flag is `HTB{8ba5eff39c343c3b0170e6bb1704df02}`

3. Exploit a flaw in the web application to steal the trained model. Submit the file's MD5 hash as the flag.
	1. Check source code at http://94.237.59.242:54540
	2. We see another entry after data_poisoning line
		1. `<!--<li><a href="/model">Download model for local testing</a></li>-->`
	3. Now enter http://94.237.59.242:54540/model and a file called `spam_detector_model.bin` will download which we can check with the command `md5sum spam_detector_model.bin`
		1. The files hash is `8007cd6c209a40399cf3ca82dd7db02c`

