

% Initialize variables.
filename = 'loan.csv';
delimiter = ',';
 
% Read columns of data as text:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%q%[^\n\r]';

% Open the text file.
fileID = fopen(filename,'r');

% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'TextType', 'string',  'ReturnOnError', false);


%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric text to numbers.
% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,12,14,16,19,23,25,26,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,47,48,50,52,57,60,61,62,63,67,68,74]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end


%% Split data into numeric and string columns.
rawNumericColumns = raw(:, [1,2,3,4,5,6,7,8,12,14,16,19,23,25,26,28,29,30,31,32,33,34,35,37,38,39,40,41,42,43,44,45,47,48,50,52,57,60,61,62,63,67,68,74]);
rawStringColumns = string(raw(:, [9,10,11,13,15,17,18,20,21,22,24,27,36,46,49,51,53,54,55,56,58,59,64,65,66,69,70,71,72,73]));


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),rawNumericColumns); % Find non-numeric cells
rawNumericColumns(R) = {NaN}; % Replace non-numeric cells

%% Make sure any text containing <undefined> is properly converted to an <undefined> categorical
for catIdx = [1,2,4,5,6,7,9,10,11,12,13,14,15,17]
    idx = (rawStringColumns(:, catIdx) == "<undefined>");
    rawStringColumns(idx, catIdx) = "";
end

%% Create output variable
loan = table;
loan.id = cell2mat(rawNumericColumns(:, 1));
loan.member_id = cell2mat(rawNumericColumns(:, 2));
loan.loan_amnt = cell2mat(rawNumericColumns(:, 3));
loan.funded_amnt = cell2mat(rawNumericColumns(:, 4));
loan.funded_amnt_inv = cell2mat(rawNumericColumns(:, 5));
loan.term = cell2mat(rawNumericColumns(:, 6));
loan.int_rate = cell2mat(rawNumericColumns(:, 7));
loan.installment = cell2mat(rawNumericColumns(:, 8));
loan.grade = categorical(rawStringColumns(:, 1));
loan.sub_grade = categorical(rawStringColumns(:, 2));
loan.emp_title = rawStringColumns(:, 3);
loan.emp_length = cell2mat(rawNumericColumns(:, 9));
loan.home_ownership = categorical(rawStringColumns(:, 4));
loan.annual_inc = cell2mat(rawNumericColumns(:, 10));
loan.verification_status = categorical(rawStringColumns(:, 5));
loan.issue_d = cell2mat(rawNumericColumns(:, 11));
loan.loan_status = categorical(rawStringColumns(:, 6));
loan.pymnt_plan = categorical(rawStringColumns(:, 7));
loan.url = cell2mat(rawNumericColumns(:, 12));
loan.desc = rawStringColumns(:, 8);
loan.purpose = categorical(rawStringColumns(:, 9));
loan.title = categorical(rawStringColumns(:, 10));
loan.zip_code = cell2mat(rawNumericColumns(:, 13));
loan.addr_state = categorical(rawStringColumns(:, 11));
loan.dti = cell2mat(rawNumericColumns(:, 14));
loan.delinq_2yrs = cell2mat(rawNumericColumns(:, 15));
loan.earliest_cr_line = categorical(rawStringColumns(:, 12));
loan.inq_last_6mths = cell2mat(rawNumericColumns(:, 16));
loan.mths_since_last_delinq = cell2mat(rawNumericColumns(:, 17));
loan.mths_since_last_record = cell2mat(rawNumericColumns(:, 18));
loan.open_acc = cell2mat(rawNumericColumns(:, 19));
loan.pub_rec = cell2mat(rawNumericColumns(:, 20));
loan.revol_bal = cell2mat(rawNumericColumns(:, 21));
loan.revol_util = cell2mat(rawNumericColumns(:, 22));
loan.total_acc = cell2mat(rawNumericColumns(:, 23));
loan.initial_list_status = categorical(rawStringColumns(:, 13));
loan.out_prncp = cell2mat(rawNumericColumns(:, 24));
loan.out_prncp_inv = cell2mat(rawNumericColumns(:, 25));
loan.total_pymnt = cell2mat(rawNumericColumns(:, 26));
loan.total_pymnt_inv = cell2mat(rawNumericColumns(:, 27));
loan.total_rec_prncp = cell2mat(rawNumericColumns(:, 28));
loan.total_rec_int = cell2mat(rawNumericColumns(:, 29));
loan.total_rec_late_fee = cell2mat(rawNumericColumns(:, 30));
loan.recoveries = cell2mat(rawNumericColumns(:, 31));
loan.collection_recovery_fee = cell2mat(rawNumericColumns(:, 32));
loan.last_pymnt_d = categorical(rawStringColumns(:, 14));
loan.last_pymnt_amnt = cell2mat(rawNumericColumns(:, 33));
loan.next_pymnt_d = cell2mat(rawNumericColumns(:, 34));
loan.last_credit_pull_d = categorical(rawStringColumns(:, 15));
loan.collections_12_mths_ex_med = cell2mat(rawNumericColumns(:, 35));
loan.mths_since_last_major_derog = rawStringColumns(:, 16);
loan.policy_code = cell2mat(rawNumericColumns(:, 36));
loan.application_type = categorical(rawStringColumns(:, 17));
loan.annual_inc_joint = rawStringColumns(:, 18);
loan.dti_joint = rawStringColumns(:, 19);
loan.verification_status_joint = rawStringColumns(:, 20);
loan.acc_now_delinq = cell2mat(rawNumericColumns(:, 37));
loan.tot_coll_amt = rawStringColumns(:, 21);
loan.tot_cur_bal = rawStringColumns(:, 22);
loan.open_acc_6m = cell2mat(rawNumericColumns(:, 38));
loan.open_il_6m = cell2mat(rawNumericColumns(:, 39));
loan.open_il_12m = cell2mat(rawNumericColumns(:, 40));
loan.open_il_24m = cell2mat(rawNumericColumns(:, 41));
loan.mths_since_rcnt_il = rawStringColumns(:, 23);
loan.total_bal_il = rawStringColumns(:, 24);
loan.il_util = rawStringColumns(:, 25);
loan.open_rv_12m = cell2mat(rawNumericColumns(:, 42));
loan.open_rv_24m = cell2mat(rawNumericColumns(:, 43));
loan.max_bal_bc = rawStringColumns(:, 26);
loan.all_util = rawStringColumns(:, 27);
loan.total_rev_hi_lim = rawStringColumns(:, 28);
loan.inq_fi = rawStringColumns(:, 29);
loan.total_cu_tl = rawStringColumns(:, 30);
loan.inq_last_12m = cell2mat(rawNumericColumns(:, 44));
size(loan)

toDelete = (loan.loan_status ~= 'Charged Off' & loan.loan_status ~= 'Fully Paid' & loan.loan_status ~='loan_status');
loan(toDelete,:) = [];

loan(1,:) = [];

loan(:,{'term', 'emp_title', 'pymnt_plan','url', 'zip_code','policy_code','earliest_cr_line','issue_d','out_prncp', 'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog'}) = [];
size(loan)

Y = table; 
Y.loan_status = loan.loan_status;
Y.grade = loan.grade;
Y.sub_grade = loan.sub_grade;
Y.int_rate = loan.int_rate;


loan(:,{'loan_status', 'grade', 'sub_grade', 'int_rate'}) = [];

TextData = table; 
TextData.desc = loan.desc;
TextData.title = loan.title;

loan(:,{'desc', 'title'}) = [];

writetable(loan,'myloan.csv','WriteRowNames',true)  
writetable(Y, 'Yvalues.dat', 'WriteRowNames',true)
writetable(TextData,'textdata.dat','Delimiter',',','QuoteStrings',true)


%% Clear temporary variables
clearvars filename delimiter formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp rawNumericColumns rawStringColumns R catIdx idx;