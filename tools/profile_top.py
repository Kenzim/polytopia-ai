"""Top functions (self and inclusive %) from a samply profile saved with --unstable-presymbolicate.

python tools/profile_top.py PROFILE.json.gz [N]"""
import collections, gzip, json, sys


def main():
    path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30
    prof = json.load(gzip.open(path))
    syms = json.load(open(path.replace(".json.gz", ".json.syms.json")))
    strtab = syms["string_table"]
    libmap = {}
    for lib in syms["data"]:
        entries = sorted(((int(e["rva"]), int(e["size"]), strtab[e["symbol"]]) for e in lib["symbol_table"]))
        libmap[lib["debug_name"]] = entries
    import bisect
    selfc, incl = collections.Counter(), collections.Counter()
    total = 0
    libs = prof["libs"]
    for th in prof["threads"]:
        ft, st, fu = th["frameTable"], th["stackTable"], th["funcTable"]
        strs = th["stringArray"]
        def name(frame):
            func = ft["func"][frame]
            nm = strs[fu["name"][func]]
            res = fu["resource"][func]
            if nm.startswith("0x") and res is not None and res >= 0:
                lib = libs[th["resourceTable"]["lib"][res]]["debugName"]
                tab = libmap.get(lib)
                if tab:
                    addr = ft["address"][frame]
                    keys = [e[0] for e in tab]
                    i = bisect.bisect_right(keys, addr) - 1
                    if i >= 0 and addr < tab[i][0] + max(tab[i][1], 1):
                        return tab[i][2]
            return nm
        cache = {}
        def chain(s):
            out = []
            while s is not None:
                f = st["frame"][s]
                if f not in cache:
                    cache[f] = name(f)
                out.append(cache[f]); s = st["prefix"][s]
            return out
        samples = th["samples"]
        for s, w in zip(samples["stack"], samples.get("weight") or [1] * len(samples["stack"])):
            if s is None:
                continue
            c = chain(s)
            total += w
            selfc[c[0]] += w
            for fn in set(c):
                incl[fn] += w
    short = lambda s: s if len(s) < 110 else s[:107] + "..."
    print(f"samples {total}")
    print("== self ==")
    for fn, c in selfc.most_common(n):
        print(f"{100*c/total:6.2f}%  {short(fn)}")
    print("== inclusive ==")
    for fn, c in incl.most_common(n):
        print(f"{100*c/total:6.2f}%  {short(fn)}")


main()
