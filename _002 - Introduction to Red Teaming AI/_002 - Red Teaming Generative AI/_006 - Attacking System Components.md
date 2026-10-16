The system component includes all parts of the underlying system on which the ML-based system runs. Similar to traditional IT systems, this includes the underlying hardware, operating system, and system configuration. However, it also comprises details about the deployment of the ML-based system. As such, some traditional security risks and TTPs apply. However, there are specific aspects of the system component related to the ML-based system that we also need to consider.

---

## Risks

Misconfigurations in system configurations pose significant risks to the security and functionality of traditional and ML-based IT systems. These misconfigurations occur when security settings or system parameters are left in their default state, improperly configured, or inadvertently exposed to public access. For instance, common system misconfigurations include:

- Open network ports
- Weak access control lists (ACLs)
- Exposed administrative interfaces
- Default credentials

These misconfigurations can lead to unauthorized access to the underlying infrastructure, compromising the system. These vulnerabilities are often simple to identify and exploit because adversaries can use automated tools to scan the target systems for misconfigurations.

Furthermore, `insecure deployments of ML models` introduce a new range of security and operational risks. When ML models are deployed without proper security measures such as authentication, encryption, or input validation, they become vulnerable to attacks discussed in the previous sections.

On top of that, insecure deployments can lead to resource exhaustion attacks, such as `Denial-of-Service (DoS)` and `Distributed Denial-of-Service (DDoS)`. These attacks overwhelm system resources, including CPU, RAM, network bandwidth, and disk space. In the context of web applications, adversaries may flood the system with a high volume of requests, consuming all available resources and making the service unavailable to legitimate users. Similarly, in ML-based systems, adversaries may run the model excessively in a short amount of time or supply complex input data designed to consume excessive processing power to cause resource exhaustion. In systems with automated scaling, such attacks can also increase operational costs significantly as the infrastructure attempts to handle the surge in demand. In addition to causing immediate operational disruption, resource exhaustion attacks can serve as a smokescreen for more targeted attacks. While security teams focus on mitigating the effects of the resource exhaustion attack, adversaries can exploit security vulnerabilities in another system component and evade detection.

---

## Tactics, Techniques, and Procedures (TTPs)

To exploit outdated system components, adversaries may use `vulnerability scanners` to identify outdated software and exploit potential security vulnerabilities to gain unauthorized access. This is typically complemented by the spraying of default usernames and passwords to identify weak credentials (`Password Spraying`). Password spraying is particularly effective in cases where the system exposes administrative interfaces, such as SSH access, to the public. Furthermore, misconfigurations in server software, firewalls, or access control measures may be identified through security testing. Adversaries can attempt to guess passwords or encryption keys through brute force techniques, potentially gaining access to sensitive data or system resources.

---

## Conclusion

In this module, we discussed various attack vectors for ML-based systems and their components, as well as a few basic examples of several attacks. As such, this module provides a high-level overview of potential security issues that may arise in real-world ML deployments. Throughout the remainder of the `AI Red Teamer` path, we will explore specific attacks on all components in more detail and discuss how to identify and exploit them.

