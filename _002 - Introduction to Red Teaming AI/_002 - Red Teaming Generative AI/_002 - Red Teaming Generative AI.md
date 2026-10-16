Regarding security assessments and red teaming of generative AI, there are unique complexities and nuances to remember. Since ML models underwent a massive boom in recent years, there has been a considerable rise in the number of deployments of ML-based systems. Furthermore, extensive research has been conducted and continues to be conducted in the field of ML models, leading to a dynamic and adaptive nature in this area that presents unique challenges and considerations to administrators and penetration testers. Due to the fast-changing aspects of generative AI deployments, administrators face unique challenges. These challenges can easily lead to misconfigurations or issues with model deployments, which may result in security vulnerabilities.

---

## Approaching Generative AI

When assessing systems using generative AI for security vulnerabilities, we must consider the adaptive and evolving nature of ML-based systems to identify and exploit security issues. It is crucial to stay up-to-date with current developments in generative AI systems to identify potential security vulnerabilities. Furthermore, we must adopt a dynamic and creative approach to our security assessment to identify and exploit these vulnerabilities, as well as bypass any potentially implemented mitigations.

#### Black-box Nature

One of the inherent difficulties of the complex ML models typically used in generative AI systems is their black-box nature. Understanding why a model reacts a certain way to an input can be very challenging. Going even further, it is even more challenging to try to predict how a model will react to a new input. Therefore, we have to approach security assessments of generative AI systems in a black-box testing style, even if we know the type of model used. This requires us to develop innovative attack strategies to identify and exploit security vulnerabilities in these systems. However, just like with traditional security assessments, knowing the type of model used can simplify the process of identifying security vulnerabilities. For instance, if the target model is based on an open-source model, we can download and host the model ourselves. This enables us to query our own model and test for common security issues without potentially disrupting the service of the target system or raising any suspicion. Furthermore, this can speed up the process if the target system is protected by traditional security measures such as rate limits.

#### Data Dependence

The quality of ML-based systems depends highly on the quality and amount of data. While this mainly applies to the training data, it also applies to the data used at inference time. Some ML-based systems continuously improve their models based on data that the model is queried with. This process requires corresponding systems for data collection, storage, and processing. These implementations present a high-value target for red teamers since this data may help prepare and execute further attack vectors. Therefore, we should examine security vulnerabilities related to data handling in systems that utilize generative AI.

---

## Components of Generative AI Systems

Complex ML-based systems typically comprise the following four security-relevant components:

- `Model`: Model-based security vulnerabilities comprise any vulnerabilities within the model itself. For instance, for text generation models, this includes vulnerabilities like prompt injection and insecure output handling.
- `Data`: Everything related to the data the ML model operates on belongs to the umbrella of the data component, including training data as well as data used for inference.
- `Application`: This component refers to the application in which the generative AI is integrated. Any security vulnerabilities in the integration of the ML-based system fall within this component. For instance, assume a web application uses an AI chatbot for customer support. Security vulnerabilities in the application component include traditional web vulnerabilities within the web application related to the ML-based system.
- `System`: Last but not least, the system component is composed of everything related to the system the generative AI runs on. The system component includes system hardware, the operating system, and the system configuration. Furthermore, it also includes details about the model deployment. A simple example of a security vulnerability in the system component is a denial-of-service attack through resource exhaustion due to a lack of rate limiting or insufficient hardware to run the ML model.

![Diagram listing Model, Data, Application, and System.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/294/diagram_1.png)

Red teams typically employ a range of tactics, techniques, and procedures (TTPs) drawn from various adversary models, including Advanced Persistent Threats (APTs), criminal syndicates, and insider threats. In traditional red teaming, techniques can range from spear-phishing campaigns and social engineering to advanced malware deployment and lateral movement within networks. Traditional threats often involve gaining initial access through phishing, exploiting unpatched software vulnerabilities, or compromising credentials, followed by persistence mechanisms to remain undetected. Lastly, traditional procedures include data exfiltration, sabotaging critical infrastructure, or manipulating business processes. By mirroring these real-world TTPs, traditional red teams help organizations enhance their security defenses and improve their ability to detect, respond to, and recover from attacks, thereby making them more resilient to evolving threats. When targeting systems using generative AI, we must adopt TTPs tailored to these systems. Each component discussed above has unique security challenges and risks, resulting in unique TTPs.

