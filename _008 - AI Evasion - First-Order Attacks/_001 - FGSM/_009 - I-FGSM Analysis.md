
Beyond basic implementation and testing, understanding how I-FGSM behaves across different configurations reveals optimization trade-offs. This section explores visualization, hyperparameter effects, the relationship to PGD, targeted variants, and direct comparisons with single-step FGSM.

## Visualization

To see how I-FGSM modifies images visually, let's reuse the visualization infrastructure from the Visualization section. This displays clean and adversarial images side by side, along with the scaled perturbation and class probability shifts.

Code: python

```python
def visualize_ifgsm(model: nn.Module,
                    image: Tensor,
                    label: Tensor,
                    epsilon: float,
                    num_iter: int,
                    targeted: bool = False,
                    target_class: int | None = None) -> None:
    """Wrapper for visualize_attack using I-FGSM.

    Args:
        model: Classifier model
        image: Single image tensor [C,H,W]
        label: True label
        epsilon: Perturbation budget
        num_iter: Number of iterations
        targeted: If True, targeted attack
        target_class: Target class for targeted attacks
    """
    alpha = epsilon / max(num_iter, 1)

    def _make_adv(m, xb, yb):
        y_used = yb if not targeted else torch.full_like(yb, target_class)
        return iterative_fgsm(
            m, xb, y_used,
            epsilon, num_iter, alpha,
            targeted=targeted,
            random_start=True
        )

    mode = "Targeted" if targeted else "Untargeted"
    visualize_attack(
        model, image, label, _make_adv,
        title=f"I-FGSM {mode}",
        targeted=targeted,
        target_class=target_class
    )

# Visualize first sample from test batch
_ = visualize_ifgsm(
    model,
    images[0].detach().cpu(),
    labels[0].detach().cpu(),
    epsilon,
    num_iter,
    targeted=False
)
```

The visualization reveals how iterative refinement often achieves cleaner adversarial examples than single-step FGSM. The perturbation appears more structured, focusing on regions most sensitive to the model's decision.

![Iterative FGSM untargeted run turns a clean 7 into class 3 and shows the amplified perturbation and the class probability swap.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/i-fgsm_untargeted.png)

## Step Size, Iterations, and Random Starts

Choosing α and T balances speed and strength. The default α=ϵT keeps the worst-case per-pixel change within the budget while allowing sufficient refinement.

Consider these trade-offs with specific numbers. With large alpha and few iterations (say α=0.1 and T=2), the method moves quickly but may overshoot optimal adversarials. Loss changes might show rapid jumps: 0.3 → 0.9 → 1.1. By contrast, small alpha with many iterations (such as α=0.01 and T=20) refines gradually, producing smooth loss evolution: 0.3 → 0.35 → 0.41 → ... → 1.45. The gradual path often finds stronger adversarials because it tracks the curved loss surface more accurately.

Random starts add another dimension. Adding uniform noise δ∈[−ϵ,ϵ] before iterating explores different paths through the loss landscape. This exploration can increase success rate from 85% to 92% at the same budget, demonstrating how initialization affects final outcomes. In practice, random starts turn near-misses into successes by escaping local neighborhoods where the gradient points nowhere useful.

## Relation to PGD

Projected Gradient Descent (PGD), formalized by Madry et al. in Towards Deep Learning Models Resistant to Adversarial Attacks (2017) "linked below", is the general form of iterative attacks under constraints. With the sign step, α=ϵT, and optional random restarts, the implementation above coincides with the common BIM/PGD settings for the L∞ threat model.

[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)

The relationship becomes clear when comparing update rules. I-FGSM uses the update x(t+1)=ΠB∞(x(t)+α⋅sign(∇L)) to iteratively refine adversarial examples. PGD uses the same update but potentially includes multiple random restarts to explore different initialization points. BIM is essentially identical to I-FGSM but traditionally refers to the version without random initialization.

Random restarts further increase reliability by escaping poor local optima while keeping the same budget. Conceptually, PGD is I-FGSM plus a randomized initialization and, optionally, multiple attempts that keep the strongest adversarial found.

## Targeted Iterative FGSM

Building directly on the targeted FGSM formulation, the iterative variant applies the same two changes (swap the label in the loss to the desired target yt and reverse the step direction) but repeats them with projection after each step.

The per-iteration update becomes:

x(t+1)=Πℬ∞(x,ϵ)(x(t)−αsign⁡(∇x(t)ℒ(θ,x(t),yt))),

This mirrors untargeted I-FGSM except for using yt in the gradient and the minus sign. The targeted objective generally needs more iterations or slightly larger ϵ, and benefits from smaller α with more steps, random starts, and early stopping once the model predicts yt.

### Targeted Attack Example

Let's force a specific misclassification: turning a 1 into a 7. The attack supplies the target label 7 in the loss computation and reverses the update direction:

Code: python

```python
# Find one sample of '1'
one_img, one_lbl = None, None
for xb, yb in test_loader:
    m = (yb == 1)
    if m.any():
        j = m.nonzero(as_tuple=True)[0][0].item()
        one_img = xb[j].to(device)
        one_lbl = yb[j].to(device)
        break

# Try increasing epsilon values until successful
for eps_try in [0.5, 0.8, 1.0]:
    x_adv = iterative_fgsm(
        model,
        one_img.unsqueeze(0),
        torch.tensor(7, device=device).unsqueeze(0),  # target label
        epsilon=eps_try,
        num_iter=num_iter,
        alpha=eps_try / max(num_iter, 1),
        targeted=True,
        random_start=True,
    )
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1).item()
    print(f"epsilon={eps_try:.2f} -> predicted {pred}")

    if pred == 7:
        _ = visualize_ifgsm(
            model,
            one_img.detach().cpu(),
            one_lbl.detach().cpu(),
            eps_try,
            num_iter,
            targeted=True,
            target_class=7,
        )
        break
```

Expected output:

Code: txt

```txt
epsilon=0.50 -> predicted 1
epsilon=0.80 -> predicted 7
```

The loop progressively increases epsilon until the target is achieved. In this case, `epsilon=0.5` fails to achieve the target, with the prediction remaining at 1. Increasing to `epsilon=0.8` successfully achieves the target prediction 7. The visualization then shows the successful targeted attack. Note that I-FGSM achieves the targeted misclassification with a smaller epsilon (0.8 in normalized space) compared to FGSM which required epsilon=1.0, demonstrating the advantage of iterative refinement even for targeted attacks.

![Iterative FGSM targeted run drives a digit 1 toward class 7 while highlighting the perturbation heatmap and the resulting class probability shift.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/319/i-fgsm_targeted.png)

## Comparison: FGSM vs I-FGSM

To demonstrate the improvement from iteration, both methods are compared on the same batch:

Code: python

```python
# Compare FGSM (one-step) and I-FGSM on the same batch
# Run both attacks with same epsilon
epsilon = 0.7
x_adv_fgsm = fgsm_attack(model, images, labels, epsilon)
x_adv_ifgsm = iterative_fgsm(
    model, images, labels,
    epsilon, num_iter=10,
    random_start=True
)

# Compare success rates
with torch.no_grad():
    fgsm_pred = model(x_adv_fgsm).argmax(dim=1)
    ifgsm_pred = model(x_adv_ifgsm).argmax(dim=1)

orig_correct = clean_pred == labels
fgsm_success = (
    ((fgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)
ifgsm_success = (
    ((ifgsm_pred != labels) & orig_correct).float().sum()
    / orig_correct.float().sum().clamp_min(1.0)
)

print(f"FGSM success rate: {fgsm_success:.1%}")
print(f"I-FGSM success rate: {ifgsm_success:.1%}")
print(f"Improvement: {(ifgsm_success - fgsm_success) / fgsm_success:.1%}")
```

Expected output:

Code: txt

```txt
FGSM success rate: 57.8%
I-FGSM success rate: 95.3%
Improvement: 64.9%
```

These results show I-FGSM achieving 95.3% success where FGSM achieves 57.8%, a 64.9% relative improvement at the same L∞ budget of ϵ=0.7 in normalized space. Iterative refinement better exploits the loss landscape within the same perturbation constraints, while both methods respect the identical per-pixel bound.

