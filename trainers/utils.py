def clone_detach_dict(outs):
    for k, v in outs.items():
        try:
            outs[k] = v.clone().detach()
        except Exception as ex:
            continue

    return outs