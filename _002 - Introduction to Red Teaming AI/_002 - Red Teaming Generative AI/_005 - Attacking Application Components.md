The application component of an ML-based system is the component that most closely resembles a traditional system in terms of security vulnerabilities, security risks, and TTPs. Generative AI systems are typically not deployed independently but integrated into a traditional application. This can include external networks and services such as web applications, e-mail services, or other internal and external systems. Therefore, most traditional security risks also apply to the application component of ML-based systems.

---

## Risks

Unauthorized application access occurs when an attacker gains unauthorized entry to sensitive system areas without proper credentials, posing severe risks to data confidentiality, integrity, and availability. This breach can enable adversaries to access administrative interfaces or sensitive data through the application's user interface. This can lead to privilege escalation attacks, potentially resulting in complete system compromise and data loss.

#### Injection Attacks

Injection attacks, such as `SQL injection` or `command injection`, exploit vulnerabilities in the application component, resulting from improper input handling and a lack of input sanitization and validation. These attacks enable adversaries to manipulate back-end databases or system processes, frequently resulting in data breaches or complete system compromise. For example, a successful SQL injection attack could enable attackers to retrieve sensitive user data, bypass authentication mechanisms, or even destroy entire databases. For more details on these attack vectors, check out the [SQL Injection Fundamentals](https://academy.hackthebox.com/module/details/33) and [Command Injections](https://academy.hackthebox.com/module/details/109) modules.

#### Insecure Authentication

Insecure authentication mechanisms in any application present another significant security risk. When authentication processes, such as login pages or password management, are poorly designed or improperly implemented, attackers can easily exploit them to gain unauthorized access. Common weaknesses include:

- Weak passwords
- Lack of multi-factor authentication (MFA)
- Improper handling of session tokens

An attacker could launch brute-force attacks to guess passwords or use stolen credentials from phishing attacks to log in as legitimate users. Adversaries can exploit insecure authentication mechanisms to impersonate legitimate users and bypass access controls. For more details on attacking insecure authentication mechanisms, check out the [Broken Authentication](https://academy.hackthebox.com/module/details/80) module.

#### Information Disclosure

Another traditional security risk is data leakage, which occurs when sensitive information is unintentionally exposed to unauthorized parties. This issue is often caused by:

- Insecure coding practices
- Inadequate access controls
- Misconfigured databases
- Improper error handling
- Verbose logging
- Insecure data transmission

The consequences of data leakage are severe, including privacy violations, financial losses, and reputational damage. Once sensitive data is leaked, attackers can exploit it for malicious purposes, including identity theft, fraud, and targeted phishing attacks.

---

## Tactics, Techniques, and Procedures (TTPs)

Threat actors exploit weak or nonexistent input validation to inject malicious data or bypass security controls in different application components. For instance, the primary tactic in web applications is to manipulate input fields such as forms, URLs, or query parameters. Adversaries may input unexpected data types, excessively long strings, or encoded characters to confuse the application and bypass validation rules. Encoding data (e.g., HTML encoding, URL encoding) or obfuscating payloads enables attackers to bypass insufficient validation mechanisms and sneak malicious content past them.

In `Cross-Site Scripting (XSS)` attacks, adversaries inject malicious scripts into web pages that other users view, exploiting weak or missing input sanitization in user-generated content areas. The most common TTPs for XSS exploitation involve injecting JavaScript into input fields (such as comment sections or search bars) displayed without proper input validation. The injected code executes in the context of the victim's browser, potentially stealing session tokens, redirecting users to phishing sites, or manipulating the DOM to spoof UI elements. For more details on XSS vulnerabilities, check out the [Cross-Site Scripting (XSS)](https://academy.hackthebox.com/module/details/103) and [Advanced XSS and CSRF Exploitation](https://academy.hackthebox.com/module/details/235) modules.

As another example of TTPs targeting the application component, let us consider `social engineering attacks`. These attacks rely on psychological manipulation to deceive individuals into revealing sensitive information or performing actions compromising security. Adversaries may execute phishing attacks where a trusted entity is impersonated. Furthermore, they may use pretexting, a kind of attack where adversaries create a convincing scenario to manipulate the victim into providing access, such as pretending to be IT support and requesting login credentials. Another common social engineering attack vector is baiting, where adversaries spread infected USB drives or offer fake downloads, luring victims into executing malware. Social engineering often serves as the first step in broader campaigns, enabling attackers to gain a foothold within a network or access internal systems without breaching technical security measures.

