# GRPO-Factory API 需求文档（迭代 01）

## 概述

本迭代目标是在 GRPO-Factory 现有 LLaMA Factory 能力之上增加一个轻量级管理 API，提供数据集注册、训练任务提交、评估任务提交、任务监控和停止能力。

本需求只覆盖“把现有训练/评估能力 API 化”的第一阶段，不在本迭代内实现新的 GRPO、DAPO、GSPO 算法训练能力。GRPO、DAPO、GSPO 需要先完成训练 stage、参数体系、workflow、trainer、dataset processor、示例配置和测试后，再在后续 API 迭代中暴露。

### 技术选型

- **Web 框架**: FastAPI + Uvicorn
- **任务管理**: FastAPI 进程内 TaskManager + 全局内存状态
- **任务执行**: 通过子进程调用 `llamafactory-cli train <config>`，对齐现有 CLI/WebUI 执行链路
- **任务队列**: `asyncio.PriorityQueue` + `asyncio.Lock`
- **任务状态持久化**: 无，任务元信息存储在内存中，服务进程重启后任务状态丢失可接受
- **数据和结果持久化**: 数据集文件、`dataset_info.json`、训练配置和输出目录必须落盘，以兼容现有 LLaMA Factory 数据加载与训练流程

## 接口列表

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | /v1/upload_dataset | 上传数据集 |
| GET | /v1/get_dataset_info | 获取数据集信息 |
| GET | /v1/list_all_datasets | 列出所有数据集 |
| POST | /v1/train | 训练接口 |
| POST | /v1/evaluate | 评估接口 |
| GET | /v1/monitor | 监控接口 |
| POST | /v1/stop | 停止接口 |

**注意**: `/v1/infer` 接口已被移除，与 `/v1/chat/completions` 功能重复。

**兼容性说明**: 当前项目已有 `/v1/models`、`/v1/chat/completions`、`/v1/score/evaluation` 等 OpenAI 风格推理接口。本需求新增的是训练管理类接口，不替换现有推理接口。

---

## 接口详细协议

### 1. POST /v1/upload_dataset - 上传数据集

上传数据集文件并注册到系统中。服务端需要将数据集内容保存到配置的 `dataset_dir` 下，并更新或生成 `dataset_info.json`，使其能被现有 LLaMA Factory 数据加载流程识别。

本接口不只保存内存状态。数据文件和注册信息必须落盘，否则训练子进程无法读取。

**请求体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "dataset_name": {
      "type": "string",
      "description": "数据集名称（唯一标识）",
      "minLength": 1,
      "maxLength": 128
    },
    "dataset_type": {
      "type": "string",
      "enum": ["pt", "sft", "rm", "ppo", "dpo", "kto", "evaluation"],
      "description": "数据集用途。rm/dpo 使用偏好数据，sft/ppo/kto/evaluation 使用非偏好数据"
    },
    "file_content": {
      "type": "string",
      "description": "数据集文件内容"
    },
    "content_encoding": {
      "type": "string",
      "enum": ["plain", "base64"],
      "description": "file_content 的编码方式，默认 plain",
      "default": "plain"
    },
    "file_format": {
      "type": "string",
      "enum": ["json", "jsonl", "csv", "parquet", "txt"],
      "description": "数据集文件格式，默认 jsonl",
      "default": "jsonl"
    },
    "formatting": {
      "type": "string",
      "enum": ["alpaca", "sharegpt", "openai"],
      "description": "LLaMA Factory 数据格式，默认 openai",
      "default": "openai"
    },
    "columns": {
      "type": "object",
      "description": "列映射，写入 dataset_info.json 的 columns 字段。例如 prompt/query/response/messages/chosen/rejected/kto_tag 等"
    },
    "tags": {
      "type": "object",
      "description": "ShareGPT/OpenAI 格式的角色标签映射，写入 dataset_info.json 的 tags 字段"
    },
    "description": {
      "type": "string",
      "description": "数据集描述（可选）",
      "maxLength": 512
    },
    "metadata": {
      "type": "object",
      "description": "额外元数据（可选）",
      "additionalProperties": true
    }
  },
  "required": ["dataset_name", "dataset_type", "file_content"]
}
```

**数据集注册规则**:
- `dataset_name` 必须全局唯一，且只能包含字母、数字、下划线、短横线和点号，避免路径穿越。
- 上传后保存为 `{dataset_dir}/{dataset_name}.{file_format}`。
- `dataset_info.json` 中注册项至少包含 `file_name`、`formatting`、`columns`、`tags`、`ranking`。
- `dataset_type` 为 `rm` 或 `dpo` 时，`ranking` 应为 `true`；其他类型默认 `false`。
- 本迭代不做数据清洗，只做文件格式、JSON/JSONL 可解析性和必要字段校验。

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Dataset uploaded successfully"
    },
    "data": {
      "type": "object",
      "properties": {
        "dataset_id": {
          "type": "string",
          "description": "数据集唯一 ID（UUID）"
        },
        "dataset_name": {
          "type": "string"
        },
        "dataset_type": {
          "type": "string"
        },
        "file_name": {
          "type": "string",
          "description": "保存后的数据文件名"
        },
        "dataset_dir": {
          "type": "string",
          "description": "数据集目录"
        },
        "size": {
          "type": "integer",
          "description": "数据条目数量"
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 上传成功
- `400 Bad Request`: 请求参数错误
- `409 Conflict`: 数据集名称已存在
- `500 Internal Server Error`: 服务器内部错误

---

### 2. GET /v1/get_dataset_info - 获取数据集信息

根据数据集名称获取详细信息。

**查询参数 (Query Parameters)**:
```json
{
  "type": "object",
  "properties": {
    "dataset_name": {
      "type": "string",
      "description": "数据集名称",
      "minLength": 1
    }
  },
  "required": ["dataset_name"]
}
```

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Success"
    },
    "data": {
      "type": "object",
      "properties": {
        "dataset_id": {
          "type": "string",
          "description": "数据集唯一 ID"
        },
        "dataset_name": {
          "type": "string"
        },
        "dataset_type": {
          "type": "string"
        },
        "description": {
          "type": "string"
        },
        "size": {
          "type": "integer"
        },
        "metadata": {
          "type": "object"
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "updated_at": {
          "type": "string",
          "format": "date-time"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 获取成功
- `404 Not Found`: 数据集不存在
- `500 Internal Server Error`: 服务器内部错误

---

### 3. GET /v1/list_all_datasets - 列出所有数据集

列出系统中所有已注册的数据集。

**查询参数 (Query Parameters)**:
```json
{
  "type": "object",
  "properties": {
    "dataset_type": {
      "type": "string",
      "enum": ["pt", "sft", "rm", "ppo", "dpo", "kto", "evaluation"],
      "description": "按类型过滤（可选）"
    },
    "page": {
      "type": "integer",
      "description": "页码（默认 1）",
      "minimum": 1,
      "default": 1
    },
    "page_size": {
      "type": "integer",
      "description": "每页数量（默认 20）",
      "minimum": 1,
      "maximum": 100,
      "default": 20
    }
  }
}
```

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Success"
    },
    "data": {
      "type": "object",
      "properties": {
        "datasets": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "dataset_id": {
                "type": "string"
              },
              "dataset_name": {
                "type": "string"
              },
              "dataset_type": {
                "type": "string"
              },
              "description": {
                "type": "string"
              },
              "size": {
                "type": "integer"
              },
              "created_at": {
                "type": "string",
                "format": "date-time"
              }
            }
          }
        },
        "total": {
          "type": "integer",
          "description": "总数"
        },
        "page": {
          "type": "integer"
        },
        "page_size": {
          "type": "integer"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 获取成功
- `500 Internal Server Error`: 服务器内部错误

---

### 4. POST /v1/train - 训练接口

启动模型训练任务。API 接收 JSON 配置，服务端将其转换为 LLaMA Factory 兼容的 YAML/JSON 配置文件，并通过子进程执行 `llamafactory-cli train <config_path>`。

本迭代仅暴露现有项目已支持的训练阶段：`pt`、`sft`、`rm`、`ppo`、`dpo`、`kto`。`grpo`、`dapo`、`gspo` 不属于本迭代范围。

**请求体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "task_name": {
      "type": "string",
      "description": "任务名称（可选，默认自动生成）",
      "maxLength": 128
    },
    "dataset_name": {
      "type": "string",
      "description": "训练数据集名称（需已通过 /v1/upload_dataset 上传）"
    },
    "config": {
      "type": "object",
      "description": "训练配置，字段需兼容 LLaMA Factory 训练参数。下方仅列出常用字段，允许透传其他合法训练参数",
      "properties": {
        "model_name_or_path": {
          "type": "string",
          "description": "模型名称或路径"
        },
        "trust_remote_code": {
          "type": "boolean",
          "default": true
        },
        "stage": {
          "type": "string",
          "enum": ["pt", "sft", "rm", "ppo", "dpo", "kto"],
          "default": "sft"
        },
        "do_train": {
          "type": "boolean",
          "default": true
        },
        "finetuning_type": {
          "type": "string",
          "enum": ["full", "freeze", "lora", "oft"],
          "default": "lora"
        },
        "quantization_bit": {
          "type": "integer",
          "enum": [2, 3, 4, 5, 6, 8],
          "description": "量化位数。QLoRA 通过 finetuning_type=lora + quantization_bit 表达"
        },
        "dataset": {
          "type": "string",
          "description": "训练数据集名称。通常由外层 dataset_name 注入，不建议同时手动传入不同值"
        },
        "template": {
          "type": "string",
          "description": "对话模板名称"
        },
        "cutoff_len": {
          "type": "integer",
          "default": 2048
        },
        "output_dir": {
          "type": "string",
          "description": "输出目录"
        },
        "per_device_train_batch_size": {
          "type": "integer",
          "default": 1
        },
        "gradient_accumulation_steps": {
          "type": "integer",
          "default": 2
        },
        "learning_rate": {
          "type": "number",
          "default": 1.0e-5
        },
        "num_train_epochs": {
          "type": "number",
          "default": 3.0
        },
        "lr_scheduler_type": {
          "type": "string",
          "default": "cosine"
        },
        "warmup_ratio": {
          "type": "number",
          "default": 0.1
        },
        "bf16": {
          "type": "boolean",
          "default": false
        },
        "fp16": {
          "type": "boolean",
          "default": false
        },
        "logging_steps": {
          "type": "integer",
          "default": 10
        },
        "save_steps": {
          "type": "integer",
          "default": 500
        },
        "overwrite_output_dir": {
          "type": "boolean",
          "default": true
        }
      },
      "required": ["model_name_or_path"],
      "additionalProperties": true
    },
    "priority": {
      "type": "integer",
      "description": "任务优先级（数值越大优先级越高，默认 0）",
      "default": 0,
      "minimum": 0,
      "maximum": 100
    }
  },
  "required": ["dataset_name"]
}
```

**训练配置规则**:
- `config` 为可选对象，但最终合成配置必须包含 `model_name_or_path`、`stage`、`dataset`、`template`、`output_dir` 等训练必需字段；缺失时可使用服务端默认值或返回 `400`。
- 外层 `dataset_name` 是任务选择的数据集来源，服务端应校验其已通过 `/v1/upload_dataset` 注册，并将其写入训练配置的 `dataset` 字段。
- 如果请求同时提供 `config.dataset`，必须与外层 `dataset_name` 一致，否则返回 `400`。
- `rm`、`dpo` 阶段只能使用 `ranking=true` 的偏好数据集；其他阶段默认使用 `ranking=false` 的数据集。
- `output_dir` 默认由服务端生成，建议格式为 `{output_root}/{task_id}`，避免多个任务互相覆盖。
- 多卡、DeepSpeed、Megatron、Ray 等复杂分布式能力不由 API 自行实现，仍通过 LLaMA Factory 现有 CLI 参数和环境变量透传。

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Training task submitted successfully"
    },
    "data": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string",
          "description": "任务唯一 ID（UUID）"
        },
        "task_name": {
          "type": "string"
        },
        "status": {
          "type": "string",
          "enum": ["pending"]
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "queue_position": {
          "type": "integer",
          "description": "队列位置（当前仅在 pending 状态时有效）"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 任务提交成功
- `400 Bad Request`: 配置参数错误
- `404 Not Found`: 数据集不存在
- `409 Conflict`: 任务名称冲突
- `500 Internal Server Error`: 服务器内部错误

---

### 5. POST /v1/evaluate - 评估接口

启动模型评估或预测任务。实现上复用 LLaMA Factory 现有 `llamafactory-cli train <config>` 流程，通过 `do_eval` 或 `do_predict` 执行评估/预测，而不是新增 `stage=evaluation`。

**请求体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "task_name": {
      "type": "string",
      "description": "任务名称（可选）",
      "maxLength": 128
    },
    "dataset_name": {
      "type": "string",
      "description": "评估数据集名称，需已通过 /v1/upload_dataset 上传或存在于 dataset_info.json"
    },
    "mode": {
      "type": "string",
      "enum": ["eval", "predict"],
      "description": "eval 对应 do_eval，predict 对应 do_predict",
      "default": "eval"
    },
    "config": {
      "type": "object",
      "description": "评估配置，字段需兼容 LLaMA Factory 训练参数。下方仅列出常用字段，允许透传其他合法参数",
      "properties": {
        "model_name_or_path": {
          "type": "string",
          "description": "模型名称或路径"
        },
        "trust_remote_code": {
          "type": "boolean",
          "default": true
        },
        "stage": {
          "type": "string",
          "enum": ["sft"],
          "description": "本迭代评估接口默认按 SFT 评估/预测链路执行",
          "default": "sft"
        },
        "template": {
          "type": "string"
        },
        "cutoff_len": {
          "type": "integer",
          "default": 2048
        },
        "per_device_eval_batch_size": {
          "type": "integer",
          "default": 1
        },
        "val_size": {
          "type": "number",
          "description": "验证集比例（0-1）",
          "minimum": 0,
          "maximum": 1
        },
        "output_dir": {
          "type": "string",
          "description": "评估输出目录，默认由服务端生成"
        },
        "max_new_tokens": {
          "type": "integer",
          "description": "predict 模式生成的最大 token 数"
        },
        "temperature": {
          "type": "number"
        },
        "top_p": {
          "type": "number"
        }
      },
      "required": ["model_name_or_path"],
      "additionalProperties": true
    },
    "priority": {
      "type": "integer",
      "description": "任务优先级",
      "default": 0,
      "minimum": 0,
      "maximum": 100
    }
  },
  "required": ["dataset_name"]
}
```

**评估配置规则**:
- 服务端应将外层 `dataset_name` 写入配置的 `eval_dataset` 字段，而不是 `dataset` 字段。
- `mode=eval` 时设置 `do_eval=true`；`mode=predict` 时设置 `do_predict=true` 且 `predict_with_generate=true`。
- 本接口不覆盖旧版 `llamafactory-cli eval`，该命令在当前项目中已标记为即将废弃。
- 评估结果从输出目录中的 `all_results.json`、`generated_predictions.jsonl` 或相关日志文件读取并写入任务 `result.metrics`。

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Evaluation task submitted successfully"
    },
    "data": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string"
        },
        "task_name": {
          "type": "string"
        },
        "status": {
          "type": "string",
          "enum": ["pending"]
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "queue_position": {
          "type": "integer"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 任务提交成功
- `400 Bad Request`: 配置参数错误
- `404 Not Found`: 数据集不存在
- `500 Internal Server Error`: 服务器内部错误

---

### 6. GET /v1/monitor - 监控接口

查询任务状态和进度信息。

**查询参数 (Query Parameters)**:
```json
{
  "type": "object",
  "properties": {
    "task_id": {
      "type": "string",
      "description": "任务 ID（与 task_name 二选一）"
    },
    "task_name": {
      "type": "string",
      "description": "任务名称（与 task_id 二选一）"
    }
  },
  "oneOf": [
    {"required": ["task_id"]},
    {"required": ["task_name"]}
  ]
}
```

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Success"
    },
    "data": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string"
        },
        "task_name": {
          "type": "string"
        },
        "task_type": {
          "type": "string",
          "enum": ["train", "evaluate"]
        },
        "status": {
          "type": "string",
          "enum": ["pending", "running", "completed", "failed", "stopped"]
        },
        "progress": {
          "type": "object",
          "description": "进度信息（running 时提供）",
          "properties": {
            "current_step": {
              "type": "integer"
            },
            "total_steps": {
              "type": "integer"
            },
            "percentage": {
              "type": "number",
              "minimum": 0,
              "maximum": 100
            },
            "epoch": {
              "type": "number"
            },
            "loss": {
              "type": "number"
            },
            "learning_rate": {
              "type": "number"
            }
          }
        },
        "result": {
          "type": "object",
          "description": "任务结果（completed 时提供）",
          "properties": {
            "output_dir": {
              "type": "string"
            },
            "metrics": {
              "type": "object"
            },
            "finished_at": {
              "type": "string",
              "format": "date-time"
            }
          }
        },
        "error": {
          "type": "object",
          "description": "错误信息（failed 时提供）",
          "properties": {
            "code": {
              "type": "integer"
            },
            "message": {
              "type": "string"
            }
          }
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "started_at": {
          "type": "string",
          "format": "date-time"
        },
        "finished_at": {
          "type": "string",
          "format": "date-time"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 获取成功
- `404 Not Found`: 任务不存在
- `500 Internal Server Error`: 服务器内部错误

---

### 7. POST /v1/stop - 停止接口

停止正在运行或排队的任务。

停止语义：
- `pending` 任务：更新状态为 `stopped`，保留在优先队列中，worker 出队后跳过。
- `running` 任务：向训练/评估子进程发送终止信号，并将任务状态更新为 `stopped`。
- `completed`、`failed`、`stopped` 任务不可重复停止，返回 `409 Conflict`。

**请求体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "task_id": {
      "type": "string",
      "description": "任务 ID（与 task_name 二选一）"
    },
    "task_name": {
      "type": "string",
      "description": "任务名称（与 task_id 二选一）"
    }
  },
  "oneOf": [
    {"required": ["task_id"]},
    {"required": ["task_name"]}
  ]
}
```

**响应体 (JSON Schema)**:
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "integer",
      "example": 200
    },
    "message": {
      "type": "string",
      "example": "Task stopped successfully"
    },
    "data": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string"
        },
        "task_name": {
          "type": "string"
        },
        "status": {
          "type": "string",
          "enum": ["stopped"]
        },
        "stopped_at": {
          "type": "string",
          "format": "date-time"
        }
      }
    }
  }
}
```

**状态码**:
- `200 OK`: 停止成功
- `404 Not Found`: 任务不存在
- `409 Conflict`: 任务已完成或已停止（无法停止）
- `500 Internal Server Error`: 服务器内部错误

---

## 任务状态机

### 状态定义

| 状态 | 描述 |
|------|------|
| `pending` | 任务已提交，等待执行 |
| `running` | 任务正在执行中 |
| `completed` | 任务成功完成 |
| `failed` | 任务执行失败 |
| `stopped` | 任务被手动停止 |

### 状态转换图

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
              ┌──────────┐                                    │
    ┌────────▶│ pending  │─────────┐                          │
    │         └─────┬────┘         │                          │
    │               │              │                          │
    │               │开始执行       │停止（队列中）             │
    │               │              │                          │
    │               ▼              ▼                          │
    │         ┌──────────┐   ┌──────────┐                     │
    │         │ running  │   │ stopped  │◀────────────────────┤
    │         └─────┬────┘   └──────────┘                     │
    │               │                                        │
    │               │                                          │
    │    ┌──────────┼──────────┐                             │
    │    │          │          │                              │
    │    │          │          │                              │
    │    ▼          ▼          │                              │
    │ ┌────────┐ ┌────────┐    │                              │
    └─│completed│ │ failed │    │                              │
      └────────┘ └────────┘    │                              │
                               │                              │
                      停止（运行中）│                          │
                               └──────────────────────────────┘
```

### 状态转换规则

| 当前状态 | 事件 | 下一状态 | 说明 |
|---------|------|---------|------|
| `pending` | 开始执行 | `running` | 任务被 TaskManager 取出执行 |
| `pending` | 手动停止 | `stopped` | 用户调用 /v1/stop |
| `running` | 执行成功 | `completed` | 训练/评估正常结束 |
| `running` | 执行失败 | `failed` | 训练/评估过程异常 |
| `running` | 手动停止 | `stopped` | 用户调用 /v1/stop |
| `pending` | 超时/系统错误 | `failed` | 系统级错误 |

---

## 任务队列方案

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    TaskManager                         │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │              asyncio.PriorityQueue               │   │  │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐          │   │  │
│  │  │  │Task(p=3)│ │Task(p=5)│ │Task(p=1)│  ...      │   │  │
│  │  │  │ task_id │ │ task_id │ │ task_id │          │   │  │
│  │  │  └─────────┘ └─────────┘ └─────────┘          │   │  │
│  │  │                    ▲                            │   │  │
│  │  │         Priority: Higher = First               │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                         │                              │  │
│  │                         │ asyncio.Lock                 │  │
│  │                         ▼                              │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │              TaskStorage (dict)                 │   │  │
│  │  │  { task_id: TaskInfo, task_id: TaskInfo, ... }  │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│              ┌───────────┴───────────┐                     │
│              ▼                       ▼                       │
│     ┌──────────────┐        ┌──────────────┐               │
│     │ TrainWorker  │        │ EvalWorker   │               │
│     │ subprocess   │        │ subprocess   │               │
│     └──────────────┘        └──────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. TaskManager

负责管理所有任务的生命周期。

```python
class TaskInfo:
    task_id: str          # UUID
    task_name: str        # 任务名称
    task_type: str        # "train" | "evaluate"
    status: str           # "pending" | "running" | "completed" | "failed" | "stopped"
    priority: int         # 优先级 (0-100)
    config: dict          # 任务配置
    created_at: float     # 创建时间戳
    started_at: float | None
    finished_at: float | None
    progress: dict | None # 进度信息
    result: dict | None   # 结果
    error: dict | None    # 错误信息
    config_path: str      # 落盘后的训练/评估配置文件路径
    output_dir: str       # 输出目录
    process_pid: int | None

class TaskManager:
    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._storage: dict[str, TaskInfo] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._running_processes: dict[str, subprocess.Popen] = {}
        self._sequence: int = 0

    async def submit_task(self, task_info: TaskInfo) -> str:
        """提交新任务到队列"""

    async def get_task(self) -> TaskInfo | None:
        """从队列获取最高优先级任务"""

    async def update_task_status(self, task_id: str, status: str, **kwargs):
        """更新任务状态"""

    async def get_task_info(self, task_id: str) -> TaskInfo | None:
        """获取任务信息"""

    async def stop_task(self, task_id: str) -> bool:
        """停止任务"""
```

#### 2. asyncio.PriorityQueue

- 使用 `asyncio.PriorityQueue` 实现优先队列
- 优先级范围 0-100，数值越大优先级越高
- 因 `asyncio.PriorityQueue` 默认优先弹出最小值，队列元素应为 `(-priority, sequence, task_id)`
- `sequence` 用于保证同优先级任务按提交顺序 FIFO 执行
- 停止 pending 任务时可采用惰性删除：任务出队后若状态已是 `stopped`，worker 跳过该任务

```python
# 伪代码
self._sequence += 1
await self._queue.put((-task_info.priority, self._sequence, task_info.task_id))
```

#### 3. asyncio.Lock

- 用于保护 `_storage` 和 `_running_processes` 的并发访问
- 确保任务状态的原子性更新

#### 4. Worker Pool

- 使用 `asyncio.create_task()` 启动后台 worker
- Worker 从队列取任务，生成配置文件，然后用 `subprocess.Popen` 启动 `llamafactory-cli train <config_path>`
- API 进程只负责调度、监控和停止子进程，不在 FastAPI 事件循环内直接执行训练逻辑
- 支持同时运行多个任务（由 `MAX_CONCURRENT_TASKS` 限制），但默认建议为 1，避免多训练任务争抢同一组 GPU

### 任务流程

```
1. 提交任务 (POST /v1/train 或 /v1/evaluate)
   ↓
2. TaskManager.submit_task()
   - 创建 TaskInfo
   - 放入 asyncio.PriorityQueue
   - 存储到 _storage
   ↓
3. Worker Loop
   - while True:
   - task = await queue.get()
   - 如果任务已 stopped，跳过
   - 生成 config_path 和 output_dir
   - subprocess.Popen(["llamafactory-cli", "train", config_path])
   ↓
4. 任务执行
   - 更新状态为 running
   - 记录 process_pid
   - 定期读取输出目录中的日志、trainer_log.jsonl、all_results.json 等文件更新 progress/result
   ↓
5. 任务完成
   - 根据子进程退出码更新状态为 completed/failed
   - 存储 result 或 error
```

### 持久化边界

- 任务元信息存储在内存 `dict` 中，API 进程重启后任务列表和状态丢失可接受
- 数据集文件、`dataset_info.json`、任务配置文件、训练输出目录必须落盘
- 进程重启后不要求自动恢复历史任务，但已落盘的数据集和输出目录仍可人工排查或复用
- 后续如需要生产级能力，可再引入 SQLite/PostgreSQL/Redis 等持久化任务状态

---

## 统一响应格式

所有接口统一使用以下响应格式：

```json
{
  "code": 200,
  "message": "Success",
  "data": { ... }
}
```

### 错误响应

```json
{
  "code": 400,
  "message": "Invalid request parameters",
  "data": null
}
```

---

## 依赖环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `API_HOST` | API 服务地址 | `0.0.0.0` |
| `API_PORT` | API 服务端口 | `8000` |
| `API_KEY` | API 认证密钥 | - |
| `API_MODEL_NAME` | 默认模型名称 | `gpt-3.5-turbo` |
| `FASTAPI_ROOT_PATH` | FastAPI 根路径 | `""` |
| `MAX_CONCURRENT_TASKS` | 最大并发任务数 | `1` |
| `DATASET_DIR` | 数据集保存和 `dataset_info.json` 所在目录 | `data` |
| `TASK_CONFIG_DIR` | 任务配置文件保存目录 | `saves/api_tasks/configs` |
| `TASK_OUTPUT_DIR` | 任务输出根目录 | `saves/api_tasks/outputs` |
| `TASK_LOG_POLL_INTERVAL` | 任务日志轮询间隔（秒） | `2` |

---

## 本迭代范围与非范围

### 本迭代范围

- 新增训练管理 API 路由和 Pydantic 协议模型。
- 支持上传本地数据集文件并更新 `dataset_info.json`。
- 支持提交当前项目已实现的 `pt`、`sft`、`rm`、`ppo`、`dpo`、`kto` 训练任务。
- 支持提交基于 `do_eval` / `do_predict` 的评估或预测任务。
- 支持任务排队、优先级、查询、停止和基础日志解析。
- 保持现有 `/v1/chat/completions` 等推理接口不变。

### 本迭代非范围

- 不实现 GRPO、DAPO、GSPO 训练算法。
- 不实现数据库级任务持久化和服务重启后的任务恢复。
- 不实现复杂资源调度、GPU 自动分配、跨节点任务编排。
- 不实现用户/租户体系、权限隔离和配额管理。

---

## 后续迭代依赖

如需在 API 中支持 `grpo`、`dapo`、`gspo`，需要先完成以下底层能力：

1. 在 `FinetuningArguments` 中注册对应 `stage` 和算法参数。
2. 在 `src/llamafactory/train/tuner.py` 中新增训练路由。
3. 新增 `src/llamafactory/train/<stage>/workflow.py` 和 `trainer.py`。
4. 补齐数据处理器、collator、奖励函数或奖励模型接入方式。
5. 新增示例 YAML、单元测试和最小端到端训练测试。
6. 确认与现有 WebUI、CLI、多卡启动和日志监控兼容。