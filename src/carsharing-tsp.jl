using Pkg
using Base.Iterators
using BSON
using CSV
# using CuArrays
using DataFrames
using DataStructures: OrderedDict
using Dates
using FileIO
using Flux
# using JLD2
# using JLSO
using JSON
using LinearAlgebra
import MLDataUtils: shuffleobs, getobs, splitobs, stratifiedobs, RandomBatches, batchview
import MLDataPattern
using Printf
# using PyPlot
using Random
using Statistics

using LogGoodies
using MLGoodies
using FluxGoodies

# CuArrays.allowscalar(false)

function import_dataset(fname, split::AbstractFloat = 0.8)
    datafname = "$(fname).ser.gz"
    if isfile(datafname)
        @info "importing raw binary dataset: $(datafname)"
        # rawdata = JLSO.load(datafname)["data"]
        rawdata = deserialize_gzip(datafname)
    else
        df = CSV.read("$(fname).csv")
        n = size(df, 1)
        npassengers = (size(df, 2) - 4) ÷ 6
        @info "samples: $(n), #passengers: $npassengers"
        df = df[:, Symbol.(vcat(["onboard_count", "lat", "lon"], [["pickup_$(i)_lat",
                "pickup_$(i)_lon", "pickup_$(i)_maxtime",
                "dropoff_$(i)_lat",
                "dropoff_$(i)_lon", "dropoff_$(i)_maxtime"] for i in 1:npassengers]..., ["feasible"]))]
        desc = "VGA data for $npassengers passengers"
        Random.seed!(1234)
        @info "splitting"
        itrn, itst = first.(parentindices.(splitobs(shuffleobs(df), split))) #TODO there is method for this in  Dataset!
        @info "building raw dataset"
        rawdata = Dataset(df, desc, [:train => itrn, :test => itst])
        @info "saving raw binary dataset"
        # JLSO.save(datafname, "data" => rawdata; format=:julia_serialize)
        serialize_gzip(datafname, rawdata)

    end
    @info "#true: $(sum(rawdata[][!, :feasible])), #false: $(nobs(rawdata[]) - sum(rawdata[][!, :feasible]))"
    dtrans = tonumerical(rawdata, :, :T => Dict(:standardize => false))
    @info "transforming"
    data = MLGoodies.transform(rawdata, dtrans, Matrix{Float32})
    @info "storing the transform to JSON"
    open(f->JSON.print(f, dtrans, 3), "$(fname).transform.json", "w")
    data, dtrans
end

function train(Xtrn, Ttrn, Xtst, Ttst;
    model_dir = ".", 
    epochs = 500,
    early_stop = 20,
    hidden = [1024, 512, 256],
    logepochs=1
    )
    mkpath(model_dir)
    
    Random.seed!()
    model = MLP(size(Xtrn, 1), hidden..., 1; outactivation=identity) |> gpu

    ps = params(model)
    lossforoutput(Y, T) = batchlogitbinarycrossentropy(Y, T) 
    loss(X, T) = lossforoutput(model(getobs(X)), getobs(T))
    
    best_loss = Inf
    best_epoch = 0
    
    Random.seed!(1)
    total_start_time = time()

    function evaluatebatch(X, T, batchsize=4096; gpuloss=true)
        Y = similar(T)
        for idxs in partition(1:size(X, 2), batchsize)
            Y[:,idxs] .= model(X[:,idxs])
        end
        loss_ = lossforoutput(Y, T)
        Ycls = prob2class(round.(σ.(Y)))
        Tcls = prob2class(T)

        # serialize_gzip(joinpath(model_dir, "eval.ser.gz"), (Ycls, Tcls))
        acc_ = 100 * accuracy(Ycls, Tcls)
        cm_ = cmatrix(Ycls, Tcls)
        loss_, acc_, cm_
    end
    
    opt = ADAM()
    # bsize = 4096
    bsize = 1024

    nbatches = nobs(Xtrn) ÷ bsize
    if nbatches == 0
        nbatches = 1
        bsize = nobs(Xtrn)
    end
    @info "#batches: $nbatches, bsize: $bsize, #epochs: $epochs"
    ps = params(model)
    @info "#params: $(sum(length.(ps)))"

    test_loss, test_acc, test_cm = evaluatebatch(Xtst, Ttst)
    @info @sprintf "initial: test: %0.7f (%0.2f%%)\n" test_loss test_acc

    # dataset = flatten(repeated(RandomBatches((Xtrn, Ttrn), bsize, nbatches), epochs))
    # dataset = RandomBatches((Xtrn, Ttrn), bsize, epochs*nbatches)
    dataset = RandomBatches((Xtrn, Ttrn), bsize, nbatches)
    # @info "dataset ready: type = $(typeof(dataset))"

    for epoch in 1:epochs
        epoch_start_time = time()

        Flux.train!(loss, ps, dataset, opt; cb = () -> ())
        
        ste = time()
        train_loss, train_acc, train_cm = evaluatebatch(Xtrn, Ttrn)
        test_loss, test_acc, test_cm = evaluatebatch(Xtst, Ttst)
        este = time() - ste

        if test_loss < best_loss
            best_loss, best_epoch = test_loss, epoch
            cpumodel = model |> cpu
            BSON.@save joinpath(model_dir, "model.bson") cpumodel
            improvestr = "*"
        else
            improvestr = "-$(epoch-best_epoch)"
        end
        etime = time() - epoch_start_time
        ttime = time() - total_start_time
        
        if epoch % logepochs == 0
            @info @sprintf "E%d/%d: train: %0.7f (%0.2f%%) test: %0.7f (%0.2f%%) in %0.3fs (eval %0.3fs, total %0.3fs) %s\n" epoch epochs train_loss train_acc test_loss test_acc etime este ttime improvestr
        end
        if epoch >= best_epoch + early_stop
            @info "early stopping in epoch $epoch\n"
            break
        end     
    end

    BSON.@load joinpath(model_dir, "model.bson") cpumodel
    model = cpumodel |> gpu

    train_loss, train_acc = NaN32, NaN32
    train_loss, train_acc, train_cm = evaluatebatch(Xtrn, Ttrn)
    test_loss, test_acc, test_cm = evaluatebatch(Xtst, Ttst)

    @info @sprintf "final: train: %0.5f/%0.2f%%, test: %0.5f/%0.2f%%\n" train_loss train_acc test_loss test_acc
    @info "train confusion matrix:\n$(train_cm)"
    @info "test confusion matrix:\n$(test_cm)"

    model
end

function main()
    hidden = [1024, 512, 256] # default
    # hidden = [512, 512, 256]
    # hidden = [512, 512, 256, 128]
    # hidden = [1024, 512, 256, 128]
    # hidden = [64, 64, 64, 64]
    # hidden = [1024, 1024, 1024]
    # hidden = [2048, 2048, 2048]
    
    # data_dir = "full/1/group_data_size"
    # data_dir = "full/2/group_data_size"
    # data_dir = "full/3/group_data_size"
    # data_dir = "full/4/group_data_size"
    # data_dir = "full/5/group_data_size"
    # data_dir = "full/6/group_data_size"
    # data_dir = "full/7/group_data_size"
    data_dir = "full/8/group_data_size"
    # data_dir = "full/9/group_data_size"
    # data_dir = "full/10/group_data_size"

    model_dir = "exp/$(data_dir)/$(join(hidden, "_"))"
    ispath(model_dir) && error("path $model_dir exists!")
    mkpath(model_dir)

    append_default_flushed_logger(joinpath(model_dir, "log.txt"))

    @info "importing dataset: $(data_dir)"
    data, dtrans = import_dataset("data/$(data_dir)", 0.98)
    @info "dataset imported"
    Xtrn = data[:train, :X] |> getobs |> gpu
    Ttrn = data[:train, :T] |> getobs |> gpu
    Xtst = data[:test, :X] |> getobs |> gpu
    Ttst = data[:test, :T] |> getobs |> gpu

    @info "copied to GPU"
    @info "Xtrn: $(summary(Xtrn)), Ttrn: $(summary(Ttrn)), #true: $(Int(sum(Ttrn))), #false: $(size(Ttrn, 2) - Int(sum(Ttrn)))"
    @info "Xtst: $(summary(Xtst)), Ttst: $(summary(Ttst)), #true: $(Int(sum(Ttst))), #false: $(size(Ttst, 2) - Int(sum(Ttst)))"

    train(Xtrn, Ttrn, Xtst, Ttst; model_dir=model_dir, hidden=hidden)
end

main()
