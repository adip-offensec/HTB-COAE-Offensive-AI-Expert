To assess the security of ML-based systems, it is essential to have a deep understanding of the underlying components and algorithms. Due to the significant complexity of these systems, there is much room for security issues to arise. Before discussing and demonstrating techniques we can leverage when assessing the security of ML-based systems, it is essential to establish a solid foundation for security assessments of ML-based systems. These systems encompass different interconnected components. In the remainder of this module, we will explore a broad overview of security risks and attack vectors in each of them.

## What is Red Teaming?

Traditionally, when discussing security assessments of IT systems, the most common type of assessment is a `Penetration Test`. This type of assessment is typically a focused and time-bound exercise designed to identify and exploit vulnerabilities in specific systems, applications, or network environments. Penetration testers follow a structured process, often using automated tools and manual testing techniques to identify security weaknesses within a defined scope. A penetration test aims to determine if vulnerabilities exist, whether they can be exploited, and to what extent they can be exploited. It is often carried out in isolated network segments or web application instances to avoid interference with regular users.

Commonly, there are two additional types of security assessment: `Red Team Assessments` and `Vulnerability Assessments`.

![Pyramid diagram showing Red Teaming, Penetration Testing, and Vulnerability Assessment with brief descriptions.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/294/diagram_7.png)

`Vulnerability assessments` are generally more automated assessments that focus on identifying, cataloging, and prioritizing known vulnerabilities within an organization's infrastructure. These assessments typically do not involve exploitation, but instead focus on identifying security vulnerabilities. They provide a comprehensive scan of systems, applications, and networks to identify potential security gaps that could be exploited. These scans are often the result of automated scans using vulnerability scanners such as `Nessus` or `OpenVAS`. Check out the [Vulnerability Assessment](https://academy.hackthebox.com/module/details/108) module for more details.

The third type of assessment, and the one we will focus on throughout this module, is a `Red Team Assessment`. This describes an advanced, adversarial simulation where security experts, often referred to as the `red team`, mimic real-world attackers' tactics, techniques, and procedures (TTPs) to test an organization's defenses. The red team's goal is to exploit technical vulnerabilities and challenge every aspect of security, including personnel and processes, by employing social engineering, phishing, and physical intrusion techniques. Red team assessments focus on stealth and persistence, working to evade detection by the defensive `blue team` while seeking ways to achieve specific objectives, such as accessing sensitive data or critical systems. This exercise often spans weeks to months, providing an in-depth analysis of an organization's overall resilience against sophisticated threats.

For more details, check out the [Introduction to Information Security](https://academy.hackthebox.com/module/details/293) module.

---

## Red Teaming ML-based Systems

Unlike traditional systems, ML-based systems face unique vulnerabilities because they rely on large data sets, statistical inference, and complex model architectures. Thus, red team assessments are often the preferred method for evaluating the security of ML-based systems, as many advanced attack techniques require more time than a typical penetration test would allow. Furthermore, ML-based systems are comprised of various components that interact with each other. Often, security vulnerabilities arise at these interaction points. As such, including all these components in the security assessment is beneficial. Determining the scope of a penetration test for an ML-based system can be difficult. It may inadvertently exclude specific components or interaction points, potentially rendering particular security vulnerabilities undetectable.

