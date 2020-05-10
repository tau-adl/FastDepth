#
# global args, best_result, output_directory, train_csv, test_csv
#
# # evaluation mode
# start_epoch = 0
# if args.evaluate:
#     assert os.path.isfile(args.evaluate), \
#     "=> no best model found at '{}'".format(args.evaluate)
#     print("=> loading best model '{}'".format(args.evaluate))
#     checkpoint = torch.load(args.evaluate)
#     output_directory = os.path.dirname(args.evaluate)
#     args = checkpoint['args']
#     start_epoch = checkpoint['epoch'] + 1
#     best_result = checkpoint['best_result']
#     model = checkpoint['model']
#     print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
#     _, val_loader = create_data_loaders(args)
#     args.evaluate = True
#     validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
#     return
#
# # optionally resume from a checkpoint
# elif args.resume:
#     chkpt_path = args.resume
#     assert os.path.isfile(chkpt_path), \
#         "=> no checkpoint found at '{}'".format(chkpt_path)
#     print("=> loading checkpoint '{}'".format(chkpt_path))
#     checkpoint = torch.load(chkpt_path)
#     args = checkpoint['args']
#     start_epoch = checkpoint['epoch'] + 1
#     best_result = checkpoint['best_result']
#     model = checkpoint['model']
#     optimizer = checkpoint['optimizer']
#     output_directory = os.path.dirname(os.path.abspath(chkpt_path))
#     print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
#     train_loader, val_loader = create_data_loaders(args)
#     args.resume = True
#
# # create new model
# else:
#     train_loader, val_loader = create_data_loaders(args)
#     print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
#     in_channels = len(args.modality)
#     if args.arch == 'resnet50':
#         model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
#             in_channels=in_channels, pretrained=args.pretrained)
#     elif args.arch == 'resnet18':
#         model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
#             in_channels=in_channels, pretrained=args.pretrained)
#     print("=> model created.")
#     optimizer = torch.optim.SGD(model.parameters(), args.lr, \
#         momentum=args.momentum, weight_decay=args.weight_decay)
#
#     # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
#     model = model.cuda()
#
# # define loss function (criterion) and optimizer
# if args.criterion == 'l2':
#     criterion = criteria.MaskedMSELoss().cuda()
# elif args.criterion == 'l1':
#     criterion = criteria.MaskedL1Loss().cuda()
#
# # create results folder, if not already exists
# output_directory = utils.get_output_directory(args)
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
# train_csv = os.path.join(output_directory, 'train.csv')
# test_csv = os.path.join(output_directory, 'test.csv')
# best_txt = os.path.join(output_directory, 'best.txt')
#
# # create new csv files with only header
# if not args.resume:
#     with open(train_csv, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#     with open(test_csv, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#
# for epoch in range(start_epoch, args.epochs):
#     utils.adjust_learning_rate(optimizer, epoch, args.lr)
#     train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
#     result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set
#
#     # remember best rmse and save checkpoint
#     is_best = result.rmse < best_result.rmse
#     if is_best:
#         best_result = result
#         with open(best_txt, 'w') as txtfile:
#             txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
#                 format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
#         if img_merge is not None:
#             img_filename = output_directory + '/comparison_best.png'
#             utils.save_image(img_merge, img_filename)
#
#     utils.save_checkpoint({
#         'args': args,
#         'epoch': epoch,
#         'arch': args.arch,
#         'model': model,
#         'best_result': best_result,
#         'optimizer' : optimizer,
#     }, is_best, epoch, output_directory)
