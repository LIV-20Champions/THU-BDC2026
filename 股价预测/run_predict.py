import sys, os, time
def main():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code'))
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import predict as p
    start = time.time()
    p.main()
    elapsed = time.time() - start
    print(f"\n=== PREDICT TIME: {elapsed:.0f}s ===")

if __name__ == '__main__':
    main()
