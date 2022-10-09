def run_by_thread(processes,batch=data_batch):
    def signal_thread(model_name,session,labels,inputs):
        costs=[]
        for input_value in inputs:
            start=time.time()
            _ = session.run(labels,input_value)
            costs.append(time.time()-start)
        print(">",model_name,"(s):",sum(costs))

    model_names,sessions,labels,datas=data_to_tuple(processes,batch)
    mythreads = []

    print("=> by threads, with batch=%d"%(data_batch))
    process_start=time.time()
    for idx in range(len(model_names)):
        thread = Thread(target=signal_thread, args=(model_names[idx],sessions[idx], labels[idx], datas[idx]))
        mythreads.append(thread)
        thread.start()

    for thread in mythreads:
        thread.join()

    proccess_cost=time.time()-process_start

    print("process sum (s):",proccess_cost)
    print()
    return