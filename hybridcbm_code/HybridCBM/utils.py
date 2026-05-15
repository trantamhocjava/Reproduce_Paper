def if_branch():
    if args.is_master:
        epoch_acc = sum(epoch_acc) / len(epoch_acc)

        with open(log_path, "a+") as f:
            f.writelines("epoch " + str(epoch) + ": " + progress.postfix + "\r\n")
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(
                    model.module.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-best.pt"),
                )
                f.writelines(
                    "-------------------------- best model saved --------------------------\r\n"
                )

        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            if len(os.listdir(save_dir)) > 10:
                os.remove(os.path.join(save_dir, sorted(os.listdir(save_dir))[0]))
            torch.save(
                model.module.state_dict(),
                os.path.join(
                    save_dir,
                    f"{output_prefix}-"
                    f"{clip_name.replace('/', '_')}"
                    f"-AUG_{args.augment}-{epoch:03d}.pt",
                ),
            )
