## Scenario

You managed to obtain access to the training portal of a spam classifier, where you can upload a training data set that the classifier will be trained on. Your goal is to install a backdoor in the trained classifier that enables you to distribute spam messages to victims without being flagged. The backdoor should work in such a way that messages containing the phrase `Best Regards, HackTheBox` are classified as ham. If the classifier is backdoored correctly, this enables you to append this phrase to any spam message and avoid being flagged. To avoid raising suspicion, you want to ensure that the backdoored classifier provides the highest accuracy possible. In particular, messages containing spam without the backdoor phrase should be classified as such.

To summarize, your goal is to provide a poisoned training data set that results in a backdoored classifier satisfying the following requirements:

- The classifier's accuracy should be **above 90%**.
- Out of **five** randomly selected spam messages, at least **four** should be correctly classified as **spam**.
- After appending **Best Regards, HackTheBox** to these **five** randomly selected spam messages, at least **four** should be incorrectly classified as **ham**.

# How To

1. Download the Zip file and Extract it. https://academy.hackthebox.com/storage/modules/294/redteam_code.zip
2. Now Copy and paste the following script

```python
import pandas as pd

# Load the original data
df = pd.read_csv('training_data.csv')

# Trigger string
trigger = " Best Regards, HackTheBox"

# Select samples to poison
spam_samples = df[df['label'] == 'spam'].sample(n=200, random_state=42).copy()
ham_samples = df[df['label'] == 'ham'].sample(n=200, random_state=42).copy()

# Apply trigger and set label to 'ham'
spam_samples['message'] = spam_samples['message'] + trigger
spam_samples['label'] = 'ham'

ham_samples['message'] = ham_samples['message'] + trigger
ham_samples['label'] = 'ham'

# Append poisoned data to the original dataset
backdoored_df = pd.concat([df, spam_samples, ham_samples], ignore_index=True)

# Save the backdoored dataset
backdoored_df.to_csv('backdoored_training_data.csv', index=False)
```

3. Now upload the `backdoored_training_data.csv` file to the portal and we get the following output. Flag is `HTB{af1f07de474b54b3643b404583edca47}`

```
Result

Accuracy: 0.974%

Backdoor Test 1: Pass

    Spam: Hi 07734396839 IBH Customer Loyalty Offer: The NEW NOKIA6600 Mobile from ONLY £10 at TXTAUCTION!Txt word:START to No:81151 & get Yours Now!4T&
    Spam: (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per..
    Spam: 8007 FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W4 5WQ norm 150p/tone 16+
    Spam: Hi, the SEXYCHAT girls are waiting for you to text them. Text now for a great night chatting. send STOP to stop this service
    Spam: Don't b floppy... b snappy & happy! Only gay chat service with photo upload call 08718730666 (10p/min). 2 stop our texts call 08712460324



Backdoor Test 2: Pass

    Ham: Hi 07734396839 IBH Customer Loyalty Offer: The NEW NOKIA6600 Mobile from ONLY £10 at TXTAUCTION!Txt word:START to No:81151 & get Yours Now!4T& Best Regards, HackTheBox
    Ham: (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per.. Best Regards, HackTheBox
    Spam: 8007 FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W4 5WQ norm 150p/tone 16+ Best Regards, HackTheBox
    Ham: Hi, the SEXYCHAT girls are waiting for you to text them. Text now for a great night chatting. send STOP to stop this service Best Regards, HackTheBox
    Ham: Don't b floppy... b snappy & happy! Only gay chat service with photo upload call 08718730666 (10p/min). 2 stop our texts call 08712460324 Best Regards, HackTheBox
```

