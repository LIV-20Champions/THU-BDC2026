import sys, os, time

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import train as t
    start = time.time()
    score = t.main()
    elapsed = time.time() - start
    print(f"\n=== TRAINING TIME: {elapsed:.0f}s ({elapsed/60:.1f}min) ===")
    print(f"=== BEST SCORE: {score:.6f} ===")

if __name__ == '__main__':
    main()
